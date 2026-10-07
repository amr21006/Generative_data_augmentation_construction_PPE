"""Day-to-night CycleGAN (PyTorch port of the repository script Gen_AI.py).

Configuration reproduced from Gen_AI.py:
- images resized (nearest neighbour) to 700 x 1200 (H x W), random 512 x 512 crop,
  random rot90 / flips for training, pixels scaled to [-1, 1];
- generator: 7x7 conv (64) -> three stride-2 3x3 convs (128, 256, 256) -> 6 residual
  blocks (conv-ReLU-conv, no normalisation) -> three 3x3 transposed convs with
  U-Net-style concatenation skips -> 7x7 conv + tanh; instance normalisation in the
  encoder/decoder blocks; weights ~ N(0, 0.02);
- discriminator: PatchGAN, 4x4 convs (64, 128, 256, 512 stride 2; 512 stride 1),
  instance normalisation, LeakyReLU(0.2), 4x4 output conv;
- losses: binary cross-entropy adversarial loss (discriminator loss halved) and L1
  cycle-consistency loss with lambda = 10;
- Adam, learning rate 2.5e-4, beta1 = 0.5, batch size 2, 200 steps per epoch,
  up to 50 epochs, early stopping (patience 10) that restores the best weights;
- seed 7.

Two behaviours of Gen_AI.py are reproduced as they actually executed:
- its LearningRateScheduler acted on the Keras model's default optimizer, not on
  the Adam optimizer used by train_step, so the effective learning rate was constant;
- its identity term compared an image with the output of the *other* generator and
  was differentiated only with respect to the generator not involved, so it
  contributed no gradient. It is therefore omitted here (identical updates).
Batch size 2 is realised directly (--bs 2) or as two accumulated single-image steps, which is exactly
equivalent because instance normalisation has no batch dependence and the losses are
batch means.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

IMG = 512
RESIZE_HW = (700, 1200)  # overridden by --crop (resize scaled so crop/width ratio is preserved)
LAMBDA = 10.0
LR = 2.5e-4
SEED = 7


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.InstanceNorm2d) and m.affine:
        nn.init.normal_(m.weight, 0.0, 0.02)  # tfa gamma_initializer RandomNormal(0, 0.02)
        nn.init.zeros_(m.bias)


class Down(nn.Module):
    def __init__(self, cin, cout, k, s, act):
        super().__init__()
        if k == 4 and s == 1:  # TF 'same' padding for an even kernel: 1 before, 2 after
            self.pad = nn.ZeroPad2d((1, 2, 1, 2))
            p = 0
        else:
            self.pad = nn.Identity()
            p = (k - 1) // 2 if s == 1 else (k - s + 1) // 2
            if k == 3 and s == 2:
                p = 1
            if k == 4 and s == 2:
                p = 1
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False)
        self.norm = nn.InstanceNorm2d(cout, affine=True)
        self.act = act

    def forward(self, x):
        return self.act(self.norm(self.conv(self.pad(x))))


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(cin, cout, 3, 2, 1, output_padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(cout, affine=True)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(x)))


class Generator(nn.Module):
    def __init__(self, n_res=6):
        super().__init__()
        r = nn.ReLU()
        self.e1 = Down(3, 64, 7, 1, r)
        self.e2 = Down(64, 128, 3, 2, r)
        self.e3 = Down(128, 256, 3, 2, r)
        self.e4 = Down(256, 256, 3, 2, r)
        self.res = nn.Sequential(*[Res(256) for _ in range(n_res)])
        self.u1 = Up(512, 256)
        self.u2 = Up(256 + 256, 128)
        self.u3 = Up(128 + 128, 64)
        self.out = nn.Conv2d(64 + 64, 3, 7, 1, 3, bias=False)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        h = self.res(e4)
        h = self.u1(torch.cat([h, e4], 1))
        h = self.u2(torch.cat([h, e3], 1))
        h = self.u3(torch.cat([h, e2], 1))
        return torch.tanh(self.out(torch.cat([h, e1], 1)))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        a = nn.LeakyReLU(0.2)
        self.body = nn.Sequential(
            Down(3, 64, 4, 2, a), Down(64, 128, 4, 2, a), Down(128, 256, 4, 2, a),
            Down(256, 512, 4, 2, a), Down(512, 512, 4, 1, a))
        self.out = nn.Conv2d(512, 1, 4, 1, 0)

    def forward(self, x):
        return self.out(self.body(x))


def _resize_one(path):
    im = Image.open(path).convert('RGB')
    return np.asarray(im.resize((RESIZE_HW[1], RESIZE_HW[0]), Image.NEAREST))


def preload(paths):
    """Decode and resize every image once (the Gen_AI.py resize step) and keep it in RAM."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        return np.stack(list(ex.map(_resize_one, paths)))


def load_train(arr, rng):
    a = arr
    y = rng.integers(0, RESIZE_HW[0] - IMG + 1)
    x = rng.integers(0, RESIZE_HW[1] - IMG + 1)
    a = a[y:y + IMG, x:x + IMG]
    p = rng.random()
    if p > 0.8:
        a = np.rot90(a, 3)
    elif p > 0.6:
        a = np.rot90(a, 2)
    elif p > 0.4:
        a = np.rot90(a, 1)
    p = rng.random()
    if p > 0.7 and rng.random() < 0.5:
        a = a[:, ::-1]
    elif p < 0.3 and rng.random() < 0.5:
        a = a[::-1, :]
    return np.ascontiguousarray(a)  # uint8 HWC; scaled to [-1, 1] on the GPU


def to_gpu(batch, dev):
    t = torch.from_numpy(np.stack(batch)).pin_memory().to(dev, non_blocking=True)
    t = t.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    return t


def bce(logits, target_one):
    t = torch.ones_like(logits) if target_one else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, t)


def train(args):
    global IMG, RESIZE_HW
    IMG = args.crop
    RESIZE_HW = (round(700 * args.crop / 512), round(1200 * args.crop / 512))
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True
    day = preload(sorted(Path(args.data, 'day').glob('*.jpg')))
    night = preload(sorted(Path(args.data, 'night').glob('*.jpg')))
    day_paths = sorted(Path(args.data, 'day').glob('*.jpg'))
    print('preloaded', day.shape, night.shape, 'crop', IMG, flush=True)
    bs, accum = args.bs, 2 // args.bs
    G, Fn = Generator().to(dev), Generator().to(dev)          # G: day->night, Fn: night->day
    Dy, Dx = Discriminator().to(dev), Discriminator().to(dev)  # Dy judges night, Dx judges day
    for m in (G, Fn, Dy, Dx):
        m.apply(init_weights)
    optG = torch.optim.Adam(G.parameters(), LR, betas=(0.5, 0.999))
    optF = torch.optim.Adam(Fn.parameters(), LR, betas=(0.5, 0.999))
    optDy = torch.optim.Adam(Dy.parameters(), LR, betas=(0.5, 0.999))
    optDx = torch.optim.Adam(Dx.parameters(), LR, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler('cuda', enabled=False)  # fp32 training
    out = Path(args.out)
    (out / 'samples').mkdir(parents=True, exist_ok=True)
    log = []
    best = {'n2d': float('inf'), 'd2n': float('inf')}
    best_epoch, wait = 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        sums = {'gen_D2N_loss': 0.0, 'gen_N2D_loss': 0.0, 'disc_night_loss': 0.0,
                'disc_day_loss': 0.0, 'cycle_loss': 0.0}
        acc_t = torch.zeros(5, device=dev)
        for step in range(args.steps):
            for o in (optG, optF, optDy, optDx):
                o.zero_grad(set_to_none=True)
            for _ in range(accum):  # batch size 2 (directly, or as exact accumulation of 1+1)
                x = to_gpu([load_train(day[rng.integers(len(day))], rng) for _ in range(bs)], dev)
                y = to_gpu([load_train(night[rng.integers(len(night))], rng) for _ in range(bs)], dev)
                if True:
                    fake_y = G(x)
                    cyc_x = Fn(fake_y)
                    fake_x = Fn(y)
                    cyc_y = G(fake_x)
                    cyc = LAMBDA * ((x - cyc_x).abs().mean() + (y - cyc_y).abs().mean())
                    gG = bce(Dy(fake_y), True) + cyc
                    gF = bce(Dx(fake_x), True) + cyc
                # G's gradient comes from gG and F's from gF; each loss also depends on the
                # other generator through the shared cycle term, so compute them separately.
                g_params, f_params = list(G.parameters()), list(Fn.parameters())
                gradsG = torch.autograd.grad(scaler.scale(gG / accum), g_params, retain_graph=True)
                gradsF = torch.autograd.grad(scaler.scale(gF / accum), f_params)
                for p, g in zip(g_params, gradsG):
                    p.grad = g if p.grad is None else p.grad + g
                for p, g in zip(f_params, gradsF):
                    p.grad = g if p.grad is None else p.grad + g
                if True:
                    dY = 0.5 * (bce(Dy(y), True) + bce(Dy(fake_y.detach()), False))
                    dX = 0.5 * (bce(Dx(x), True) + bce(Dx(fake_x.detach()), False))
                scaler.scale((dY + dX) / accum).backward()
                acc_t += torch.stack([gG, gF, dY, dX, cyc]).detach().float() / accum
            for o in (optG, optF, optDy, optDx):
                scaler.step(o)
            scaler.update()
        vals = (acc_t / args.steps).tolist()
        row = dict(zip(['gen_D2N_loss', 'gen_N2D_loss', 'disc_night_loss', 'disc_day_loss', 'cycle_loss'], vals))
        row['epoch'] = epoch
        row['elapsed_min'] = (time.time() - t0) / 60
        log.append(row)
        print(json.dumps(row), flush=True)
        # Early stopping as in Gen_AI.py: both generator losses must improve.
        if row['gen_N2D_loss'] < best['n2d'] and row['gen_D2N_loss'] < best['d2n']:
            best = {'n2d': row['gen_N2D_loss'], 'd2n': row['gen_D2N_loss']}
            best_epoch, wait = epoch, 0
            torch.save({'G': G.state_dict(), 'F': Fn.state_dict(), 'epoch': epoch}, out / 'best.pt')
        else:
            wait += 1
        torch.save({'G': G.state_dict(), 'F': Fn.state_dict(), 'epoch': epoch}, out / 'last.pt')
        save_samples(G, Fn, day_paths[:4], out / 'samples' / f'epoch{epoch:03d}.jpg', dev)
        with open(out / 'cyclegan_log.json', 'w') as f:
            json.dump({'log': log, 'best_epoch': best_epoch, 'stopped_epoch': epoch,
                       'config': {'crop': IMG, 'resize_hw': RESIZE_HW, 'batch': 2, 'batch_per_pass': bs,
                                  'lr': LR, 'beta1': 0.5, 'lambda': LAMBDA, 'seed': SEED, 'max_epochs': args.epochs,
                                  'steps_per_epoch': args.steps, 'n_day': int(len(day)), 'n_night': int(len(night)),
                                  'precision': 'fp32', 'gpu': torch.cuda.get_device_name(0),
                                  'torch': torch.__version__}}, f, indent=1)
        if wait >= 10:
            print(f'early stopping at epoch {epoch}; best epoch {best_epoch}', flush=True)
            break
    (out / 'DONE').write_text(json.dumps({'best_epoch': best_epoch, 'stopped_epoch': epoch}))  # completion marker


@torch.no_grad()
def translate_array(G, a, dev, long_side=640):
    """Translate an HxWx3 uint8 array; returns the translated image at the resized size."""
    h, w = a.shape[:2]
    s = long_side / max(h, w)
    nh, nw = max(8, round(h * s)), max(8, round(w * s))
    im = np.asarray(Image.fromarray(a).resize((nw, nh), Image.BICUBIC), dtype=np.float32)
    ph, pw = (-nh) % 8, (-nw) % 8
    t = torch.from_numpy(im.transpose(2, 0, 1) / 127.5 - 1.0).unsqueeze(0).to(dev)
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    o = G(t)
    o = o[:, :, :nh, :nw].float().clamp(-1, 1)
    return ((o[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).round().astype(np.uint8)


def save_samples(G, Fn, paths, dst, dev):
    G.eval(); Fn.eval()
    rows = []
    for p in paths:
        a = np.asarray(Image.open(p).convert('RGB'))
        n = translate_array(G, a, dev, 384)
        c = translate_array(Fn, n, dev, 384)
        base = np.asarray(Image.fromarray(a).resize((n.shape[1], n.shape[0])))
        rows.append(np.concatenate([base, n, c], 1))
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    Image.fromarray(np.concatenate(rows, 0)).save(dst, quality=90)
    G.train(); Fn.train()


def translate_dir(args):
    dev = 'cuda'
    ck = torch.load(args.weights, map_location=dev)
    G = Generator().to(dev)
    G.load_state_dict(ck['G'])
    G.eval()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    for i, p in enumerate(files):
        a = np.asarray(Image.open(p).convert('RGB'))
        o = translate_array(G, a, dev, args.long_side)
        Image.fromarray(o).save(dst / (p.stem + '_n2.jpg'), quality=95)
        if i % 200 == 0:
            print(i, len(files), flush=True)
    print('translated', len(files), 'epoch', ck.get('epoch'))


@torch.no_grad()
def panels(args):
    """Qualitative panels: original | photometric darkening | CycleGAN G(x) | cycled F(G(x))."""
    dev = 'cuda'
    ck = torch.load(args.weights, map_location=dev)
    G, Fn = Generator().to(dev), Generator().to(dev)
    G.load_state_dict(ck['G']); Fn.load_state_dict(ck['F'])
    G.eval(); Fn.eval()
    dst = Path(args.dst); dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.names.split(','):
        p = next(Path(args.src).glob(name + '.*'))
        a = np.asarray(Image.open(p).convert('RGB'))
        n = translate_array(G, a, dev, args.long_side)
        c = translate_array(Fn, n, dev, args.long_side)
        base = np.asarray(Image.fromarray(a).resize((n.shape[1], n.shape[0]), Image.BICUBIC))
        # photometric darkening equal to the strongest HSV value reduction of the detector's default augmentation (gain 0.6)
        import cv2
        hsv = cv2.cvtColor(base, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 2] *= 0.6
        dark = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        for tag, im in (('original', base), ('photometric', dark), ('cyclegan', n), ('cycled', c)):
            Image.fromarray(im).save(dst / f'{p.stem}_{tag}.png')
        rows.append(np.concatenate([base, dark, n, c], 1))
        diff = np.abs(base.astype(np.float32) - c.astype(np.float32)).mean()
        print(p.stem, 'mean |x - F(G(x))| =', round(float(diff), 2), flush=True)
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 6), (0, wmax - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    Image.fromarray(np.concatenate(rows, 0)).save(dst / 'panels_grid.jpg', quality=92)


def stats(args):
    """Mean luminance of image folders (0-255), for describing the translated images."""
    out = {}
    for d in args.dirs.split(','):
        vals = []
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            im = Image.open(p).convert('L'); im.thumbnail((320, 320))
            vals.append(float(np.asarray(im, dtype=np.float32).mean()))
        v = np.array(vals)
        out[d] = dict(n=len(v), median=float(np.median(v)), q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)))
        print(d, out[d], flush=True)
    Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--epochs', type=int, default=50)
    t.add_argument('--steps', type=int, default=200)
    t.add_argument('--bs', type=int, default=1, choices=[1, 2])
    t.add_argument('--crop', type=int, default=512)
    r = sub.add_parser('translate')
    r.add_argument('--weights', required=True)
    r.add_argument('--src', required=True)
    r.add_argument('--dst', required=True)
    r.add_argument('--long_side', type=int, default=640)
    q = sub.add_parser('panels')
    q.add_argument('--weights', required=True)
    q.add_argument('--src', required=True)
    q.add_argument('--dst', required=True)
    q.add_argument('--names', required=True, help='comma-separated image stems')
    q.add_argument('--long_side', type=int, default=640)
    st = sub.add_parser('stats')
    st.add_argument('--dirs', required=True)
    st.add_argument('--json', required=True)
    a = ap.parse_args()
    {'train': train, 'translate': translate_dir, 'panels': panels, 'stats': stats}[a.cmd](a)
