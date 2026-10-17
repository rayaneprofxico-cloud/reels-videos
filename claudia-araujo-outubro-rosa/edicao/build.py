#!/usr/bin/env python3
"""Monta o Reels da Vereadora Cláudia Araújo a partir do discurso original."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2
from PIL import Image, ImageDraw

S = sys.argv[1]
STEP = sys.argv[2] if len(sys.argv) > 2 else "all"
SRC = f"{S}/download/video.mp4"
W = f"{S}/work"
PD = f"{W}/pieces"
GD = f"{W}/gfx"
OUT = f"{S}/out"
for d in (PD, GD, OUT):
    os.makedirs(d, exist_ok=True)
MODEL = f"{S}/models/sh.rnnn"
FPS = 30
TAIL = 2.6
ZOOM = {"W": 1.0, "M": 1.15, "C": 1.30}

# (in, out, zoom, segment, contiguous-with-previous)
PIECES = [
    (165.36, 171.16, "W", 1, False), (171.40, 172.88, "C", 1, False), (172.88, 174.93, "M", 1, True),
    (151.27, 155.62, "W", 2, False), (156.37, 160.27, "M", 2, False),
    (175.18, 176.64, "W", 3, False), (177.02, 179.80, "M", 3, False), (180.35, 185.08, "C", 3, False),
    (188.84, 191.71, "M", 4, False), (191.71, 196.82, "W", 4, True), (197.22, 201.31, "C", 4, False),
    (251.79, 253.21, "M", 5, False), (255.41, 258.96, "C", 5, False),
    (276.27, 278.69, "W", 6, False), (279.44, 280.78, "C", 6, False), (281.76, 286.31, "M", 6, False),
    (44.25, 45.84, "W", 7, False), (46.14, 48.79, "C", 7, False),
    (336.93, 342.13, "M", 8, False), (343.98, 345.76, "W", 8, False),
    (374.58, 378.14, "C", 9, False), (378.38, 380.98, "M", 9, False),
    (384.13, 387.88, "W", 9, False), (387.88, 390.65, "C", 9, True),
]

# ---------------------------------------------------------------- timeline
def build_timeline():
    tl, t_out, prev = [], 0.0, None
    for a, b, z, seg, cont in PIECES:
        tin = prev["tout"] if (cont and prev) else round(a * FPS) / FPS
        n = int(round((b - tin) * FPS))
        p = dict(tin=tin, tout=tin + n / FPS, n=n, z=z, seg=seg, cont=cont, ostart=t_out)
        t_out += n / FPS
        tl.append(p)
        prev = p
    return tl, t_out

TL, SPEECH_DUR = build_timeline()
DUR = SPEECH_DUR + TAIL
ENDT = SPEECH_DUR  # início do cartão final

def m(t):
    """tempo no vídeo original -> tempo no Reels (None se foi cortado)"""
    for p in TL:
        if p["tin"] - 1e-6 <= t <= p["tout"] + 1e-6:
            return p["ostart"] + (t - p["tin"])
    return None

def seg_bounds(seg):
    ps = [p for p in TL if p["seg"] == seg]
    return ps[0]["ostart"], ps[-1]["ostart"] + ps[-1]["n"] / FPS

# ---------------------------------------------------------------- faces
FC = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PC = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

def grab(t):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", SRC, "-frames:v", "1",
                        "-vf", "scale=540:960", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(960, 540, 3)

def face_at(t):
    g = cv2.equalizeHist(cv2.cvtColor(grab(t), cv2.COLOR_BGR2GRAY))
    c = [tuple(f) for f in FC.detectMultiScale(g, 1.1, 5, minSize=(70, 70))]
    if not c:
        c = [tuple(f) for f in PC.detectMultiScale(g, 1.1, 4, minSize=(70, 70))]
        c += [(540 - x - w, y, w, h) for (x, y, w, h) in PC.detectMultiScale(cv2.flip(g, 1), 1.1, 4, minSize=(70, 70))]
    c = [f for f in c if 120 < f[1] + f[3] / 2 < 650]
    if not c:
        return None
    x, y, w, h = max(c, key=lambda f: f[2] * f[3])
    return ((x + w / 2) * 2, (y + h / 2) * 2)

def piece_face(p):
    ts = [p["tin"] + 0.25, (p["tin"] + p["tout"]) / 2, p["tout"] - 0.25]
    fs = [f for f in (face_at(t) for t in ts) if f]
    if not fs:
        return None
    return (float(np.median([f[0] for f in fs])), float(np.median([f[1] for f in fs])))

def crop_for(z, fc):
    if z == 1.0:
        return None
    cw = int(round(1080 / z / 2)) * 2
    ch = int(round(1920 / z / 2)) * 2
    fx, fy = fc if fc else (540, 740)
    cx = min(max(fx - cw / 2, 0), 1080 - cw)
    cy = min(max(fy - 0.40 * ch, 0), 1920 - ch)
    return cw, ch, int(cx), int(cy)

# ---------------------------------------------------------------- video
def enc_piece(i, p):
    vf = []
    cr = crop_for(ZOOM[p["z"]], p.get("face"))
    if cr:
        vf.append("crop=%d:%d:%d:%d" % cr)
    vf.append("scale=1080:1920:flags=lanczos,setsar=1,fps=30")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{p['tin']:.4f}", "-i", SRC, "-an",
                    "-vf", ",".join(vf), "-frames:v", str(p["n"]), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "12", "-pix_fmt", "yuv420p", "-r", "30", f"{PD}/p{i:02d}.mp4"], check=True)

def step_video():
    for p in TL:
        p["face"] = piece_face(p)
        print(f"  {p['tin']:7.2f}-{p['tout']:7.2f} z={p['z']} face={p['face']}", flush=True)
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda a: enc_piece(*a), enumerate(TL)))
    with open(f"{PD}/list.txt", "w") as f:
        for i in range(len(TL)):
            f.write(f"file 'p{i:02d}.mp4'\n")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{PD}/list.txt",
                    "-c", "copy", f"{W}/base.mp4"], check=True)

# ---------------------------------------------------------------- audio
CLEAN = ("pan=mono|c0=0.5*c0+0.5*c1,highpass=f=90,"
         f"arnndn=m={MODEL}:mix=0.8,afftdn=nr=10:nf=-40:tn=1,"
         "equalizer=f=250:t=q:w=1.2:g=-3,equalizer=f=3500:t=q:w=1.5:g=3,"
         "deesser=i=0.3,acompressor=threshold=-20dB:ratio=3:attack=8:release=120:makeup=2")

def step_audio():
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-vn", "-af", CLEAN, "-ar", "48000",
                    f"{W}/cleaned.wav"], check=True)
    runs = []
    for p in TL:
        if p["cont"] and runs:
            runs[-1][1] = p["tout"]
        else:
            runs.append([p["tin"], p["tout"]])
    parts, labels = [], []
    for k, (a, b) in enumerate(runs):
        d = b - a
        parts.append(f"[0:a]atrim=start={a:.5f}:end={b:.5f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d=0.012,afade=t=out:st={d - 0.03:.5f}:d=0.03[a{k}]")
        labels.append(f"[a{k}]")
    fc = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(runs)}:v=0:a=1,apad=pad_dur={TAIL}[out]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{W}/cleaned.wav", "-filter_complex", fc,
                    "-map", "[out]", "-t", f"{DUR:.4f}", f"{W}/speech.wav"], check=True)
    # ganho fixo até -14 LUFS (padrão Instagram) + limitador nos picos; o loudnorm dinâmico passava do alvo
    gain = -14.0 - loudness(f"{W}/speech.wav")
    for _ in range(3):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{W}/speech.wav", "-af",
                        f"volume={gain:.2f}dB,alimiter=limit=0.84:attack=3:release=60:level=false,"
                        "pan=stereo|c0=c0|c1=c0", "-ar", "48000", f"{W}/speech_ln.wav"], check=True)
        err = -14.0 - loudness(f"{W}/speech_ln.wav")
        if abs(err) < 0.3:
            break
        gain += err

def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    line = [l for l in r.stderr.splitlines() if l.strip().startswith("I:")][-1]
    return float(line.split()[1])

# ---------------------------------------------------------------- words
def load_words():
    wins = json.load(open(f"{W}/segwords.json"))
    words = []
    for w in wins:
        for x in w["words"]:
            words.append([x["s"], x["e"], x["w"].strip()])
    # S2: primeira passada tem o "os" e o tempo certo do "62"
    words = [x for x in words if not (150.5 <= x[0] < 156.0)]
    words += [[151.42, 151.54, "A"], [151.54, 151.72, "gente"], [151.72, 152.04, "teve"],
              [152.04, 152.60, "agora"], [152.66, 153.40, "os"], [153.80, 154.66, "62"],
              [154.66, 155.50, "feminicídios,"]]
    # S8: "pra ouvir, para acolher" (a segunda passada desalinhou)
    words = [x for x in words if not (342.4 <= x[0] < 346.0)]
    words += [[344.06, 344.22, "pra"], [344.22, 344.70, "ouvir,"], [344.94, 345.14, "para"],
              [345.18, 345.66, "acolher."]]
    fixes = {"passar": "passa", "Obrigado.": "Obrigada.", "coisa?": "coisa.", "violência,": "violência?"}
    out = []
    for s, e, t in sorted(words, key=lambda x: x[0]):
        if t == "corde":
            continue
        if t == "rosa.":
            s, t = 389.06, "cor-de-rosa."
        if t == "violência," and 278.0 < s < 278.5:
            t = "violência?"
        elif t in fixes and t != "violência,":
            t = fixes[t]
        out.append([s, e, t])
    # mantém só palavras que caem nos trechos usados; converte para o tempo do Reels
    res = []
    for s, e, t in out:
        for p in TL:
            a, b = max(s, p["tin"]), min(e, p["tout"])
            if b - a >= min(0.12, 0.5 * (e - s)) or (p["tin"] <= s < p["tout"]):
                res.append([p["ostart"] + max(s, p["tin"]) - p["tin"],
                            p["ostart"] + min(e, p["tout"]) - p["tin"], t])
                break
    res.sort(key=lambda x: x[0])
    for i in range(len(res) - 1):  # sem sobreposição
        res[i][1] = min(res[i][1], res[i + 1][0])
    return res

# ---------------------------------------------------------------- ASS
PINK = "&H00952DFF&"   # #FF2D95
LPINK = "&H00D9B3FF&"  # #FFB3D9
WINE = "&H0026064A&"   # #4A0626
KEYS = {"FEMINICÍDIO", "PALHAÇADA!", "SIM!", "MORRENDO", "GRITAR", "ACOLHIDAS", "DELEGACIAS", "APANHANDO",
        "MICROCRÉDITO", "OPORTUNIZANDO", "VEZ", "VOZ", "ACOLHER", "COR-DE-ROSA", "VIOLÊNCIA?"}

def clean(t):
    t = t.upper().replace("...", "").replace("…", "")
    return t.rstrip(",.;:")

def ts(t):
    t = max(0, t)
    h, r = divmod(t, 3600)
    mi, se = divmod(r, 60)
    return f"{int(h)}:{int(mi):02d}:{se:05.2f}"

def ev(t0, t1, style, text, layer=0):
    return f"Dialogue: {layer},{ts(t0)},{ts(t1)},{style},,0,0,0,,{text}"

RECT = "{\\p1}m 0 0 l 1080 0 1080 1920 0 1920{\\p0}"

def caption_events(words):
    evs = []
    i62 = next(i for i, w in enumerate(words) if w[2] == "62")
    special = {i62, i62 + 1}
    groups, cur = [], []

    def flush():
        nonlocal cur
        if cur:
            groups.append(("n", cur))
            cur = []

    for i, w in enumerate(words):
        t = clean(w[2])
        if i in special:
            flush()
            if i == i62:
                groups.append(("62", [words[i62], words[i62 + 1]]))
            continue
        if t == "PORTAS" and i + 1 < len(words) and clean(words[i + 1][2]) == "ABERTAS":
            flush(); groups.append(("k", [w, words[i + 1]])); special.add(i + 1); continue
        if t in KEYS:
            flush(); groups.append(("k", [w])); continue
        if cur:
            chars = sum(len(clean(x[2])) for x in cur) + len(t)
            if len(cur) >= 3 or chars > 14 or w[0] - cur[-1][1] > 0.35 or cur[-1][2][-1] in ",.!?":
                flush()
        cur.append(w)
    flush()

    for gi, (kind, ws) in enumerate(groups):
        g0 = ws[0][0]
        nxt = groups[gi + 1][1][0][0] if gi + 1 < len(groups) else ENDT
        g1 = min(nxt, ws[-1][1] + 0.45)
        if nxt - g1 < 0.5:
            g1 = nxt
        g1 = max(g1, g0 + 0.2)
        if kind == "62":
            g1 = min(nxt, ws[1][1] + 0.6)
            evs.append(ev(ws[0][0], g1, "Big", "{\\pos(540,1195)\\fscx140\\fscy140\\t(0,140,\\fscx100\\fscy100)}62"))
            evs.append(ev(ws[1][0], g1, "Key", "{\\pos(540,1395)\\fscx120\\fscy120\\t(0,110,\\fscx100\\fscy100)}FEMINICÍDIOS"))
        elif kind == "k":
            txt = " ".join(clean(x[2]) for x in ws)
            evs.append(ev(g0, g1, "Key", "{\\pos(540,1320)\\fscx125\\fscy125\\t(0,110,\\fscx100\\fscy100)}" + txt))
        else:
            for k, w in enumerate(ws):
                t0 = g0 if k == 0 else w[0]
                t1 = ws[k + 1][0] if k + 1 < len(ws) else g1
                if t1 - t0 < 0.01:
                    continue
                parts = []
                for j, x in enumerate(ws):
                    c = clean(x[2])
                    parts.append("{\\c%s}%s{\\c&H00FFFFFF&}" % (PINK, c) if j == k else c)
                pop = "\\fscx82\\fscy82\\t(0,90,\\fscx100\\fscy100)" if k == 0 else ""
                evs.append(ev(t0, t1, "Cap", "{\\pos(540,1320)%s}" % pop + " ".join(parts)))
    return evs

def card(t0, t1, lines, layer=2):
    """lines: [(y, style, text, delay)] cartões no topo com entrada deslizando"""
    out = []
    for y, style, text, dl in lines:
        a = t0 + dl
        out.append(ev(a, t1, style, "{\\move(540,%d,540,%d,0,160)\\fad(120,140)\\fscx92\\fscy92\\t(0,160,\\fscx100\\fscy100)}%s"
                      % (y + 28, y, text), layer))
    return out

def step_ass():
    words = load_words()
    json.dump(words, open(f"{W}/words_out.json", "w"), ensure_ascii=False, indent=0)
    E = []
    E += caption_events(words)

    # gancho + identificação
    E += card(0.0, 3.75, [(395, "HookWhite", "NÃO PODE FALAR", 0.0), (515, "HookPink", "SOBRE FEMINICÍDIO?", 0.12)])
    E += card(3.85, 8.6, [(372, "TagBig", "VEREADORA", 0.0), (452, "CardPink", "CLÁUDIA ARAÚJO", 0.1)])
    # S4: o que falta
    s4a, s4b = seg_bounds(4)
    E += card(m(190.25), s4b, [(380, "CardPink", "+ DELEGACIAS DA MULHER", 0.0)])
    E += card(m(193.20), s4b, [(478, "CardWhite", "+ CASAS DE ACOLHIMENTO", 0.0)])
    # S5: projeto
    s5a, s5b = seg_bounds(5)
    E += card(m(255.6), s5b, [(380, "CardPink", "PROJETO DE MICROCRÉDITO", 0.0),
                              (468, "CardWhiteS", "PARA MULHERES VÍTIMAS DE VIOLÊNCIA", 0.12)])
    # S7: missão
    s7a, s7b = seg_bounds(7)
    E += card(s7a, s7b, [(380, "CardWhite", "MEU PAPEL NA POLÍTICA", 0.0)])
    # S8: identidade de novo
    s8a, s8b = seg_bounds(8)
    E += card(s8a + 0.1, s8b, [(372, "TagBig", "VEREADORA", 0.0), (452, "CardPink", "CLÁUDIA ARAÚJO", 0.1)])

    # flash rosa nas viradas de bloco
    for seg in range(2, 10):
        a, _ = seg_bounds(seg)
        E.append(ev(a, a + 0.22, "Rect", "{\\an7\\pos(0,0)\\1a&H58&\\t(0,220,\\1a&HFF&)}" + RECT, 1))

    # selo Outubro Rosa (o lacinho é PNG; o texto vem aqui)
    E.append(ev(0, ENDT, "Tag", "{\\an4\\pos(128,268)}OUTUBRO ROSA", 3))

    # "um mundo bem mais cor-de-rosa": tela vai ficando rosa
    t_rosa = m(387.98)
    E.append(ev(t_rosa, ENDT + 0.4, "Rect", "{\\an7\\pos(0,0)\\1a&HFF&\\t(0,%d,\\1a&HA8&)}" % int((ENDT - t_rosa) * 1000) + RECT, 1))

    # cartão final
    E.append(ev(ENDT, DUR, "Rect", "{\\an7\\pos(0,0)\\1a&HFF&\\t(0,300,\\1a&H2A&)}" + RECT, 4))
    fin = [(805, "EndSmall", "CONTE COM A", 0.30), (885, "EndMid", "VEREADORA", 0.40),
           (990, "EndBig", "CLÁUDIA ARAÚJO", 0.50), (1150, "CardWhiteL", "@v.claudiaaraujo", 0.75),
           (1275, "EndSmall", "SIGA {\\fnDejaVu Sans}♥{\\fnInter ExtraBold} COMPARTILHE", 1.0)]
    for y, st, tx, dl in fin:
        E.append(ev(ENDT + dl, DUR, st, "{\\move(540,%d,540,%d,0,200)\\fad(180,0)\\fscx90\\fscy90\\t(0,200,\\fscx100\\fscy100)}%s"
                    % (y + 30, y, tx), 5))

    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Inter Black,84,&H00FFFFFF,&H00FFFFFF,&H00200516,&H96000000,0,0,0,0,100,100,0,0,1,7,3,5,0,0,0,1
Style: Key,Inter Black,100,&H00FFFFFF,&H00FFFFFF,{PINK[:-1]},{PINK[:-1]},0,0,0,0,100,100,0,0,3,16,0,5,0,0,0,1
Style: Big,Inter Black,270,{PINK[:-1]},&H00FFFFFF,&H00FFFFFF,&H96000000,0,0,0,0,100,100,-4,0,1,9,5,5,0,0,0,1
Style: CardPink,Inter Black,58,&H00FFFFFF,&H00FFFFFF,{PINK[:-1]},{PINK[:-1]},0,0,0,0,100,100,0,0,3,15,0,5,0,0,0,1
Style: CardWhite,Inter Black,58,{PINK[:-1]},&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,3,15,0,5,0,0,0,1
Style: HookPink,Inter Black,74,&H00FFFFFF,&H00FFFFFF,{PINK[:-1]},{PINK[:-1]},0,0,0,0,100,100,0,0,3,18,0,5,0,0,0,1
Style: HookWhite,Inter Black,74,{PINK[:-1]},&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,3,18,0,5,0,0,0,1
Style: CardWhiteS,Inter Black,40,{PINK[:-1]},&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,3,12,0,5,0,0,0,1
Style: CardWhiteL,Inter Black,66,{PINK[:-1]},&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,3,18,0,5,0,0,0,1
Style: TagBig,Inter ExtraBold,44,{LPINK[:-1]},&H00FFFFFF,{WINE[:-1]},&H96000000,0,0,0,0,100,100,6,0,1,5,2,5,0,0,0,1
Style: Tag,Inter ExtraBold,36,&H00FFFFFF,&H00FFFFFF,{WINE[:-1]},&H96000000,0,0,0,0,100,100,3,0,1,4,2,4,0,0,0,1
Style: Rect,Inter,20,{PINK[:-1]},{PINK[:-1]},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: EndSmall,Inter ExtraBold,52,&H00FFFFFF,&H00FFFFFF,{WINE[:-1]},&H96000000,0,0,0,0,100,100,4,0,1,4,0,5,0,0,0,1
Style: EndMid,Inter Black,64,&H00FFFFFF,&H00FFFFFF,{WINE[:-1]},&H96000000,0,0,0,0,100,100,8,0,1,5,0,5,0,0,0,1
Style: EndBig,Inter Black,108,&H00FFFFFF,&H00FFFFFF,{WINE[:-1]},&H96000000,0,0,0,0,100,100,0,0,1,7,4,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    open(f"{W}/captions.ass", "w").write(head + "\n".join(E) + "\n")

# ---------------------------------------------------------------- gráficos
def catmull(pts, n=24):
    pts = [pts[0]] + pts + [pts[-1]]
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = map(np.array, pts[i - 1:i + 3])
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(tuple(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)))
    out.append(tuple(pts[-2]))
    return out

def ribbon(path, width):
    K = 4
    im = Image.new("RGBA", (400 * K, 560 * K), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    a = [(95, 525), (150, 400), (200, 305), (262, 200), (268, 120), (200, 62)]
    b = [(200, 62), (132, 120), (138, 200), (200, 305), (250, 400), (305, 525)]
    A = [(x * K, y * K) for x, y in catmull(a)]
    B = [(x * K, y * K) for x, y in catmull(b)]
    for line in (B, A):
        d.line(line, fill=(255, 255, 255, 255), width=86 * K, joint="curve")
        d.line(line, fill=(255, 45, 149, 255), width=62 * K, joint="curve")
    im = im.resize((400, 560), Image.LANCZOS)
    im = im.crop(im.getbbox())
    h = int(im.height * width / im.width)
    im.resize((width, h), Image.LANCZOS).save(path)

def vignette(path):
    y, x = np.mgrid[0:1920, 0:1080].astype(np.float32)
    r = np.sqrt(((x - 540) / 540) ** 2 + ((y - 900) / 1050) ** 2)
    a = np.clip((r - 0.82) / 0.50, 0, 1) ** 1.6 * 0.40
    img = np.zeros((1920, 1080, 4), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 255, 45, 149
    img[..., 3] = (a * 255).astype(np.uint8)
    Image.fromarray(img, "RGBA").save(path)

def step_gfx():
    ribbon(f"{GD}/rib_small.png", 62)
    ribbon(f"{GD}/rib_big.png", 150)
    vignette(f"{GD}/vignette.png")

# ---------------------------------------------------------------- render
def step_render(out_name="reels_claudia_araujo.mp4", extra=None):
    fc = (f"[0:v]tpad=stop_mode=clone:stop_duration={TAIL},eq=contrast=1.04:saturation=1.06,format=yuv420p[b];"
          f"[b][2:v]overlay=0:0:shortest=1[v1];"
          f"color=c=0xFF2D95:s=1080x12:r=30:d={DUR:.3f}[bar];"
          f"[v1][bar]overlay=x='-w+w*t/{DUR:.3f}':y=0:eof_action=pass[v2];"
          f"[v2][3:v]overlay=x=50:y=226:shortest=1:enable='lt(t,{ENDT:.3f})'[v3];"
          f"[v3]subtitles=f={W}/captions.ass[v4];"
          f"[4:v]format=rgba,fade=t=in:st={ENDT + 0.1:.3f}:d=0.4:alpha=1[rb];"
          f"[v4][rb]overlay=x=(W-w)/2:y=470:shortest=1:enable='gte(t,{ENDT:.3f})'[vout]")
    cmd = ["ffmpeg", "-v", "error", "-stats", "-y", "-i", f"{W}/base.mp4", "-i", f"{W}/speech_ln.wav",
           "-framerate", "30", "-loop", "1", "-i", f"{GD}/vignette.png",
           "-framerate", "30", "-loop", "1", "-i", f"{GD}/rib_small.png",
           "-framerate", "30", "-loop", "1", "-i", f"{GD}/rib_big.png",
           "-filter_complex", fc, "-map", "[vout]", "-map", "1:a",
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-maxrate", "9M", "-bufsize", "18M",
           "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-r", "30", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
           "-t", f"{DUR:.3f}"]
    if extra:
        cmd += extra
    subprocess.run(cmd + [f"{OUT}/{out_name}"], check=True)

if __name__ == "__main__":
    print(f"pieces={len(TL)} speech={SPEECH_DUR:.2f}s total={DUR:.2f}s")
    steps = {"video": step_video, "audio": step_audio, "ass": step_ass, "gfx": step_gfx, "render": step_render}
    for name in (["gfx", "ass", "audio", "video", "render"] if STEP == "all" else STEP.split(",")):
        print("==", name, flush=True)
        steps[name]()
