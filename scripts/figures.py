"""Draw the README figures from the recorded results.

    python -m scripts.figures

Reads results/latency.json, results/batch_size.json, results/block_size.json and
figures/demo_trace.json (one run of scripts/demo.py on three prompts), and writes each chart
twice into figures/, as <name>.light.png and <name>.dark.png, so the README can follow the
reader's GitHub theme. It also writes figures/race.gif and figures/race.mp4, an animation
of the demo trace. Nothing here loads a model.
"""
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

# Validated with the dataviz palette checker: the first three categorical slots pass the
# colour-blind and normal-vision separation checks on all pairs in both modes. Aqua sits
# below 3:1 on the light surface, so every chart labels its series directly.
# Hand-placed labels where two points of the acceptance chart sit almost on top of each other.
LABEL_OFFSETS = {("spec_ngram", "long", "prose"): (-7, 4), ("spec_ngram", "short", "code"): (7, 2),
                 ("spec_draft", "short", "code"): (-7, -10), ("spec_draft", "long", "prose"): (-7, -10)}
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3de",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"], "neutral": "#8a8983",
              "accept": "#d8eee0", "accept_text": "#1f5138", "reject": "#f6dcd6", "reject_text": "#9a3b2e"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#33332f",
             "series": ["#3987e5", "#d95926", "#199e70"], "neutral": "#8f8e86",
             "accept": "#23412f", "accept_text": "#a8e3c1", "reject": "#4a2620", "reject_text": "#f08f7e"},
}
CATEGORIES = ["code", "prose", "repetitive"]
MARKERS = ["s", "o", "D"]
NAMES = {"spec_ngram": "Prompt lookup (no second model)", "spec_draft": "Draft model (135M)",
         "hf_assisted": "transformers assisted generation", "cached": "Plain decoding with KV cache",
         "no_cache": "Plain decoding, no cache", "spec_self": "Target drafts for itself"}


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def style(ax, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)
    ax.grid(axis="both", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def figure(t: dict, width: float, height: float, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    return fig, axes


def title(fig, t: dict, text: str, sub: str) -> None:
    fig.text(0.02, 0.97, text, color=t["text"], fontsize=14, fontweight="bold", va="top")
    fig.text(0.02, 0.905, sub, color=t["muted"], fontsize=10.5, va="top")


def save(fig, name: str, mode: str) -> None:
    fig.savefig(OUT / f"{name}.{mode}.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def speedup(mode: str) -> None:
    """End-to-end p50 speedup over cached decoding, one row per prompt condition."""
    t, data = THEMES[mode], load("results/latency.json")["conditions"]
    rows = [(cond, cat) for cond in ("long", "short") for cat in CATEGORIES]
    methods = ["spec_ngram", "spec_draft", "hf_assisted"]
    fig, ax = figure(t, 10, 5.6)
    style(ax, t)
    ax.grid(axis="y", visible=False)
    for i, (cond, cat) in enumerate(rows):
        cells = data[cond]["categories"][cat]
        base = cells["cached"]["e2e_ms"]["p50"]
        values = [base / cells[m]["e2e_ms"]["p50"] for m in methods]
        ax.plot([min(values), max(values)], [i, i], color=t["grid"], linewidth=2, zorder=1)
        for j, (m, v) in enumerate(zip(methods, values)):
            # Small vertical offsets and distinct shapes keep equal values (short prose) apart.
            ax.scatter(v, i + (j - 1) * 0.14, s=90, marker=MARKERS[j], color=t["series"][j], edgecolor=t["surface"],
                       linewidth=2, zorder=3, label=NAMES[m] if i == 0 else None)
    ax.axvline(1, color=t["neutral"], linestyle="--", linewidth=1.2, zorder=2)
    ax.text(1.0, len(rows) - 0.45, "  plain decoding = 1.0×", color=t["muted"], fontsize=9, va="center")
    ax.set_yticks(range(len(rows)), [f"{cond} prompt · {cat}" for cond, cat in rows], color=t["text"])
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.4, -0.5)
    ax.set_xlim(0.85, 2.1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.2g}×")
    ax.set_xlabel("Speedup (higher is faster)", color=t["muted"])
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3, frameon=False, fontsize=9.5,
                       handletextpad=0.3, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(t["text"])
    fig.subplots_adjust(left=0.2, right=0.97, top=0.8, bottom=0.11)
    title(fig, t, "The speedup depends on the text, not just the method",
          "End-to-end p50 time of plain cached decoding ÷ each method's, SmolLM2-1.7B on an Apple M3, 32 new tokens")
    save(fig, "speedup", mode)


def acceptance(mode: str) -> None:
    """Acceptance rate against speedup: the draft only pays off when it is believed."""
    t, data = THEMES[mode], load("results/latency.json")["conditions"]
    fig, ax = figure(t, 10, 5.6)
    style(ax, t)
    for j, m in enumerate(["spec_ngram", "spec_draft"]):
        xs, ys = [], []
        for cond in ("long", "short"):
            for cat in CATEGORIES:
                cells = data[cond]["categories"][cat]
                x, y = cells[m]["alpha"], cells["cached"]["e2e_ms"]["p50"] / cells[m]["e2e_ms"]["p50"]
                xs.append(x)
                ys.append(y)
                offset = LABEL_OFFSETS.get((m, cond, cat), (7, -3))
                ax.annotate(f"{cond} {cat}", (x, y), xytext=offset, textcoords="offset points",
                            ha="right" if offset[0] < 0 else "left", fontsize=8, color=t["muted"])
        marker = MARKERS[j]
        ax.scatter(xs, ys, s=80, marker=marker, color=t["series"][j], edgecolor=t["surface"], linewidth=2,
                   zorder=3, label=NAMES[m])
    ax.axhline(1, color=t["neutral"], linestyle="--", linewidth=1.2)
    ax.text(1.0, 1.0, "break-even ", color=t["muted"], fontsize=9, va="bottom", ha="right")
    ax.set_xlim(-0.03, 1.08)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.2g}×")
    ax.set_xlabel("Acceptance rate α (share of guessed tokens the big model kept)", color=t["muted"])
    ax.set_ylabel("Speedup", color=t["muted"])
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(t["text"])
    fig.subplots_adjust(left=0.08, right=0.97, top=0.8, bottom=0.12)
    title(fig, t, "More accepted guesses, more speed",
          "Each point is one prompt condition (long = 300–600 tokens, short = 9–167). Prompt lookup is nearly free, "
          "so it breaks even at a lower α")
    save(fig, "acceptance", mode)


def theory(mode: str) -> None:
    """Expected tokens per round, (1 - α^(k+1)) / (1 - α), with the measured α marked."""
    t, data = THEMES[mode], load("results/latency.json")["conditions"]["long"]["categories"]
    fig, ax = figure(t, 10, 5.4)
    style(ax, t)
    alphas = [i / 200 for i in range(201)]
    for j, k in enumerate([2, 4, 8]):
        ys = [k + 1 if a == 1 else (1 - a ** (k + 1)) / (1 - a) for a in alphas]
        ax.plot(alphas, ys, color=t["series"][j], linewidth=2)
        ax.text(1.01, ys[-1], f"k = {k}", color=t["text"], fontsize=10, va="center")
    for cat in CATEGORIES:
        a = data[cat]["spec_draft"]["alpha"]
        y = (1 - a ** 5) / (1 - a) if a < 1 else 5
        ax.scatter(a, y, s=70, color=t["series"][1], edgecolor=t["surface"], linewidth=2, zorder=4)
        ax.annotate(f"{cat}\nα {a:.2f}", (a, y), xytext=(-8, 10), textcoords="offset points", ha="right",
                    fontsize=9, color=t["text"])
    ax.set_xlim(0, 1.09)
    ax.set_ylim(0.8, 9.4)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("Acceptance rate α", color=t["muted"])
    ax.set_ylabel("Tokens produced per big-model pass", color=t["muted"])
    fig.subplots_adjust(left=0.08, right=0.93, top=0.8, bottom=0.12)
    title(fig, t, "Why the guesser's accuracy matters so much",
          "Expected tokens per verify pass, (1 − α^(k+1)) / (1 − α), for k guessed tokens. Dots: measured α of the 135M "
          "draft at k = 4, long prompts")
    save(fig, "theory", mode)


def batch(mode: str) -> None:
    """Throughput relative to plain decoding at the same batch size, one panel per drafter."""
    t, data = THEMES[mode], load("results/batch_size.json")
    sizes = data["settings"]["batches"]
    fig, axes = figure(t, 10, 4.8, ncols=2)
    for ax, m in zip(axes, ["spec_draft", "spec_ngram"]):
        style(ax, t)
        for j, cat in enumerate(CATEGORIES):
            cells = data["categories"][cat]
            ys = [cells[f"{m}@{b}"]["tokens_per_s_p50"] / cells[f"cached@{b}"]["tokens_per_s_p50"] for b in sizes]
            ax.plot(sizes, ys, color=t["series"][j], linewidth=2, marker="o", markersize=7,
                    markeredgecolor=t["surface"], markeredgewidth=2)
            nudge = -0.035 if (m, cat) == ("spec_ngram", "prose") else 0.0
            ax.text(sizes[-1] * 1.12, ys[-1] + nudge, cat, color=t["text"], fontsize=9.5, va="center")
        ax.axhline(1, color=t["neutral"], linestyle="--", linewidth=1.2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, [str(b) for b in sizes])
        ax.set_xlim(0.8, 16)
        ax.set_ylim(0.5, 2.0)
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.2g}×")
        ax.set_title(NAMES[m], color=t["text"], fontsize=11, loc="left")
        ax.set_xlabel("Sequences decoded together (batch size)", color=t["muted"])
    axes[0].set_ylabel("Throughput vs plain decoding", color=t["muted"])
    fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.14, wspace=0.22)
    title(fig, t, "The advantage fades as the batch grows",
          "Generated tokens per second (p50) ÷ plain cached decoding's at the same batch size; short prompts, k = 4")
    save(fig, "batch", mode)


def block(mode: str) -> None:
    """Share of reserved K/V slots that go unused, by cache block size."""
    t, waste = THEMES[mode], load("results/block_size.json")["waste"]
    fig, ax = figure(t, 10, 5)
    style(ax, t)
    for j, cond in enumerate(["long", "short"]):
        sizes = sorted(int(b) for b in waste[cond]["paged"])
        ys = [100 * waste[cond]["paged"][str(b)]["plain"]["waste_fraction"] for b in sizes]
        ax.plot(sizes, ys, color=t["series"][j], linewidth=2, marker="o", markersize=7,
                markeredgecolor=t["surface"], markeredgewidth=2)
        contiguous = 100 * waste[cond]["contiguous"]["waste_fraction"]
        ax.axhline(contiguous, color=t["series"][j], linestyle=":", linewidth=1.5)
        ax.text(1, contiguous + 1.5, f"{cond} prompts, one contiguous slab: {contiguous:.0f}%", color=t["text"],
                fontsize=9)
        ax.text(sizes[-1] * 1.15, ys[-1], f"{cond} prompts", color=t["text"], fontsize=9.5, va="center")
    ax.axvline(16, color=t["neutral"], linewidth=1.2)
    ax.text(16 * 1.08, 52, "chosen: 16", color=t["muted"], fontsize=9)
    ax.set_xscale("log", base=2)
    ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ax.set_xticks(ticks, [str(b) for b in ticks])
    ax.set_xlim(0.8, 700)
    ax.set_ylim(-2, 80)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    ax.set_xlabel("Tokens per cache block", color=t["muted"])
    ax.set_ylabel("Reserved memory left unused", color=t["muted"])
    fig.subplots_adjust(left=0.09, right=0.93, top=0.8, bottom=0.12)
    title(fig, t, "Paging the cache wastes far less memory than one big slab",
          "Unused share of K/V slots after 32 new tokens, over the benchmark prompts (309 long, 210 short)")
    save(fig, "block", mode)


def tpot(mode: str) -> None:
    """Time per output token on long code prompts, one bar per method."""
    t, cells = THEMES[mode], load("results/latency.json")["conditions"]["long"]["categories"]["code"]
    order = ["no_cache", "spec_self", "cached", "spec_draft", "spec_ngram", "hf_assisted"]
    values = [cells[m]["tpot_ms"]["p50"] for m in order]
    fig, ax = figure(t, 10, 4.6)
    style(ax, t)
    ax.grid(axis="y", visible=False)
    bars = ax.barh(range(len(order)), values, height=0.6, color=t["series"][0])
    bars[order.index("cached")].set_color(t["neutral"])
    for i, v in enumerate(values):
        ax.text(v + 8, i, f"{v:.0f} ms", va="center", fontsize=9.5, color=t["text"])
    ax.set_yticks(range(len(order)), [NAMES[m] for m in order], color=t["text"])
    ax.invert_yaxis()
    ax.set_xlim(0, 760)
    ax.set_xlabel("Milliseconds per generated token, p50 (lower is faster; grey is the baseline)", color=t["muted"])
    fig.subplots_adjust(left=0.31, right=0.97, top=0.78, bottom=0.14)
    title(fig, t, "Time per token on long code prompts",
          "Without a cache every token re-reads the whole 450-token prompt: ten times slower")
    save(fig, "tpot", mode)


# ---- the animation -----------------------------------------------------------------------

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
W, H, FPS = 1200, 640, 20


def lanes(trace: dict) -> tuple[list, list]:
    """(time, token count, is_draft list) events for plain and speculative decoding. The
    totals are the measured wall times; inside a run they are spread evenly per step,
    because the demo records only the total."""
    plain, spec = trace["results"][0], trace["results"][1]
    flags = [d for _, d in spec["segments"]]
    plain_events = [((i + 1) * plain["seconds"] / plain["tokens"], i + 1) for i in range(plain["tokens"])]
    rounds = trace["emitted_per_round"]
    per_step = spec["seconds"] / (len(rounds) + 1)
    spec_events, count = [(per_step, 1)], 1
    for i, emitted in enumerate(rounds):
        count += emitted
        spec_events.append(((i + 2) * per_step, count))
    return plain_events, spec_events, flags


def wrap(segments: list, width: int) -> list[list[tuple[str, bool]]]:
    lines, line, used = [], [], 0
    for text, flag in segments:
        for i, part in enumerate(text.split("\n")):
            if i:
                lines.append(line)
                line, used = [], 0
            while used + len(part) > width:
                cut = width - used
                line.append((part[:cut], flag))
                lines.append(line)
                line, used, part = [], 0, part[cut:]
            if part:
                line.append((part, flag))
                used += len(part)
    return lines + [line]


def race() -> None:
    t = THEMES["light"]
    trace = load("figures/demo_trace.json")["code"]
    plain_events, spec_events, flags = lanes(trace)
    segments = trace["results"][1]["segments"]
    mono = ImageFont.truetype(str(FONT_DIR / "DejaVuSansMono.ttf"), 15)
    sans = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 16)
    bold = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 19)
    small = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 13)
    char_w = mono.getlength("M")
    plain_s, spec_s = trace["results"][0]["seconds"], trace["results"][1]["seconds"]
    total = plain_s + 1.5
    frames = []
    for f in range(int(total * FPS) + 1):
        now = f / FPS
        img = Image.new("RGB", (W, H), t["surface"])
        d = ImageDraw.Draw(img)
        d.text((24, 16), "Same model, same answer, two ways to produce it", font=bold, fill=t["text"])
        d.text((24, 44), f"Prompt: {trace['prompt']}", font=small, fill=t["muted"])
        lanes_spec = [("Plain decoding: one token per pass of the 1.7B model", plain_events, False, plain_s),
                      ("Speculative: the 135M model guesses 4, the 1.7B checks them in one pass", spec_events, True,
                       spec_s)]
        for li, (label, events, colored, seconds) in enumerate(lanes_spec):
            top = 84 + li * 270
            shown = max([n for when, n in events if when <= now], default=0)
            done = now >= seconds
            d.rounded_rectangle((16, top, W - 16, top + 254), radius=12, outline=t["grid"], width=2)
            d.text((32, top + 10), label, font=sans, fill=t["text"])
            clock = f"{min(now, seconds):4.2f} s" + ("  ✓ done" if done else "")
            d.text((W - 32 - sans.getlength(clock), top + 10), clock, font=sans,
                   fill=t["accept_text"] if done else t["muted"])
            # progress bar
            d.rounded_rectangle((32, top + 38, W - 32, top + 46), radius=4, fill=t["grid"])
            frac = shown / len(segments)
            if frac:
                d.rounded_rectangle((32, top + 38, 32 + (W - 64) * frac, top + 46), radius=4,
                                    fill=t["series"][2] if colored else t["neutral"])
            lines = wrap([(text, flag and colored) for text, flag in segments[:shown]], int((W - 64) // char_w))
            for row, line in enumerate(lines[:9]):
                x, y = 32, top + 58 + row * 21
                for text, flag in line:
                    w = mono.getlength(text)
                    if flag:
                        d.rectangle((x, y - 1, x + w, y + 18), fill=t["accept"])
                    d.text((x, y), text, font=mono, fill=t["accept_text"] if flag else t["text"])
                    x += w
            d.text((32, top + 232), f"{shown} / {len(segments)} tokens", font=small, fill=t["muted"])
        d.rectangle((24, H - 22, 38, H - 10), fill=t["accept"])
        d.text((44, H - 25), "green: guessed by the small model and accepted by the big one     "
               f"result: {plain_s / spec_s:.2f}× faster, identical tokens", font=small, fill=t["muted"])
        frames.append(img)
    frames += [frames[-1]] * FPS * 2
    frames[0].save(OUT / "race.gif", save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0,
                   optimize=True)
    if shutil.which("ffmpeg"):
        tmp = OUT / "_frames"
        tmp.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(tmp / f"{i:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "%04d.png"),
                        "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", str(OUT / "race.mp4")], check=True)
        shutil.rmtree(tmp)


def main() -> None:
    plt.rcParams["font.family"] = [f for f in ("Helvetica Neue", "Arial", "DejaVu Sans")
                                   if f in {x.name for x in font_manager.fontManager.ttflist}][:1] or ["DejaVu Sans"]
    for mode in THEMES:
        for chart in (speedup, acceptance, theory, batch, block, tpot):
            chart(mode)
    race()
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir() if p.suffix in {".png", ".gif", ".mp4"})))


if __name__ == "__main__":
    main()
