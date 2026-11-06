"""Draw methods and recorded TP4 results without executing the notebooks.

Sources: TP1/TP4 code, saved TP4 reports and the published BERT base config.
Diagram blocks are schematic, not measured activations or predictions.
"""
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets'
INK, MUTED, RULE = '#222b34', '#526170', '#cbd3db'
BLUE, TEAL = '#356a94', '#2a7566'
PALE_BLUE, PALE_GREEN = '#edf3f8', '#edf5f0'
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
    'font.size': 13, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': INK, 'axes.edgecolor': RULE,
    'svg.fonttype': 'path', 'svg.hashsalt': 'ai-applications-research',
    'savefig.facecolor': 'white',
})


def label(ax, x, y, text, size=13, **kw):
    return ax.text(x, y, text, fontsize=size, ha=kw.pop('ha', 'center'),
                   va=kw.pop('va', 'center'), **kw)


def arrow(ax, points, color=INK):
    ax.add_patch(FancyArrowPatch(path=MplPath(points), arrowstyle='-|>',
                                mutation_scale=10, color=color, lw=.9))


def block(ax, x, y, w, h, title, body='', fill='white', edge=INK):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, lw=.7))
    if body:
        label(ax, x+w/2, y+22, title, 14, weight='bold')
        label(ax, x+w/2, y+(h+24)/2, body, 12.5, linespacing=1.35)
    else:
        label(ax, x+w/2, y+h/2, title, 13.5, linespacing=1.35)


def canvas(height):
    fig, ax = plt.subplots(figsize=(12, height/100))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 1200), ylim=(height, 0)); ax.axis('off')
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / f'{name}.svg', metadata={'Date': None})
    plt.close(fig)


def verify_metrics():
    """Check plotted values against saved text, not a new evaluation."""
    metrics = json.loads((ROOT / 'docs/results/tp4_metrics.json').read_text())
    cells = json.loads((ROOT / 'nlp/TP4.ipynb').read_text())['cells']
    for key, idx in [('svm', 12), ('bert', 21)]:
        text = ''.join(''.join(o.get('text', [])) for o in cells[idx].get('outputs', []))
        assert float(re.search(r'Accuracy: ([0-9.]+)', text)[1]) == metrics[key]['accuracy']
        for cls, expected in zip(metrics['classes'], metrics[key]['per_class_f1']):
            match = re.search(r'^\s*'+cls+r'\s+[\d.]+\s+[\d.]+\s+([\d.]+)', text, re.M)
            assert match and float(match[1]) == expected
        metrics[key]['macro_f1'] = float(re.search(r'macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)', text)[1])
    source = ''.join(''.join(c.get('source', [])) for c in cells)
    for token in ['max_length=128', 'num_train_epochs=1', 'learning_rate=5e-5',
                  'weight_decay=0.5', 'per_device_train_batch_size=8', 'max_df=0.9']:
        assert token in source, token
    assert metrics['train_samples'] == 660 and metrics['test_samples'] == 165
    return metrics


def cv_methods():
    fig, ax = canvas(410)
    label(ax, 25, 23, '(a) Intensity analysis and spatial operators', 17, ha='left', weight='bold')
    block(ax, 25, 48, 181, 66, 'Grayscale image', r'$I\in\{0,\ldots,255\}^{H\times W}$')
    arrow(ax, [(209, 81), (251, 81)])
    arrow(ax, [(251, 81), (251, 60), (283, 60)])
    arrow(ax, [(251, 81), (251, 109), (283, 109)])
    block(ax, 290, 39, 622, 42, '256-bin histogram → histogram equalization', fill=PALE_BLUE)
    block(ax, 290, 88, 622, 42, 'Mean / Gaussian / median filters · Sobel / Laplacian / Canny')
    arrow(ax, [(916, 60), (957, 60)])
    arrow(ax, [(916, 109), (957, 109)])
    label(ax, 1062, 60, 'Enhanced contrast', 13)
    label(ax, 1062, 109, 'Filtered / edge images', 13)
    ax.plot([25, 1175], [142, 142], color=RULE, lw=.7)
    label(ax, 25, 158, '(b) Cell segmentation and region counting', 17, ha='left', weight='bold')
    block(ax, 25, 194, 181, 76, 'Cell microscopy', 'Grayscale input')
    arrow(ax, [(209, 232), (251, 232)])
    block(ax, 258, 194, 190, 76, 'Otsu threshold', 'Binary foreground mask')
    arrow(ax, [(452, 232), (480, 232), (480, 196), (511, 196)])
    arrow(ax, [(480, 232), (480, 270), (511, 270)])
    block(ax, 518, 175, 396, 43, r'Contours: retain area $>8.5$ px$^2$', fill=PALE_BLUE)
    block(ax, 518, 236, 396, 66, 'Marker-based watershed', 'Opening → distance transform → markers', fill=PALE_GREEN)
    arrow(ax, [(918, 196), (958, 196)])
    arrow(ax, [(918, 269), (958, 269)])
    label(ax, 1062, 196, '328 retained regions', 13)
    label(ax, 1062, 269, '373 watershed regions', 13)
    ax.plot([25, 1175], [321, 321], color=RULE, lw=.7)
    label(ax, 25, 346, '(c) Local feature matching across views', 17, ha='left', weight='bold')
    label(ax, 112, 383, 'Two object views', 13.5)
    arrow(ax, [(209, 383), (251, 383)])
    label(ax, 373, 383, 'SIFT keypoints + descriptors', 13.5)
    arrow(ax, [(511, 383), (552, 383)])
    label(ax, 733, 383, r'BF k-NN ($k=2$) → ratio test ($0.75$)', 13.5)
    arrow(ax, [(922, 383), (960, 383)])
    label(ax, 1062, 383, 'Retained matches', 13.5)
    save(fig, 'computer-vision-methods')


def nlp_method():
    fig, ax = canvas(540)
    label(ax, 25, 23, '(a) Dataset preparation and held-out evaluation', 17, ha='left', weight='bold')
    block(ax, 25, 54, 190, 87, 'Medical Notes 2019', '825 notes\n5 specialty labels')
    arrow(ax, [(219, 97), (260, 97)])
    block(ax, 267, 54, 374, 87, 'Shared text preparation', 'Remove HTML · lowercase · normalize spaces\nEnglish stop words · Porter stemming')
    arrow(ax, [(645, 97), (687, 97)])
    block(ax, 694, 54, 211, 87, 'Stratified 80/20 split', 'Split seed: 42')
    arrow(ax, [(909, 97), (936, 97), (936, 72), (968, 72)])
    arrow(ax, [(936, 97), (936, 123), (968, 123)])
    block(ax, 974, 50, 201, 43, '660 training notes', fill=PALE_BLUE)
    block(ax, 974, 102, 201, 43, '165 held-out notes', fill=PALE_GREEN)
    label(ax, 25, 168, 'Fit both models on the training partition; evaluate the fitted models on the same held-out rows.', 12.5, ha='left', color=MUTED)
    ax.plot([25, 1175], [191, 191], color=RULE, lw=.7)
    label(ax, 25, 218, '(b) Representations and classification heads', 17, ha='left', weight='bold')
    label(ax, 25, 250, 'TF-IDF + LinearSVC', 14, ha='left', color=BLUE, weight='bold')
    block(ax, 25, 274, 305, 74, 'TF-IDF vectorizer', 'Fit on training text · max_df = 0.9', fill=PALE_BLUE)
    arrow(ax, [(334, 311), (370, 311)])
    label(ax, 473, 292, 'Sparse note vector', 13.5)
    label(ax, 473, 322, r'$x\in\mathbb{R}^{|V_{\mathrm{train}}|}$', 16)
    arrow(ax, [(574, 311), (614, 311)])
    block(ax, 621, 274, 267, 74, 'LinearSVC', r'$f_c(x)=w_c^{\mathsf{T}}x+b_c$', fill=PALE_BLUE)
    arrow(ax, [(892, 311), (936, 311)])
    label(ax, 1061, 293, '5 decision scores', 13.5)
    label(ax, 1061, 324, r'$\hat{y}=\operatorname{arg\,max}_c f_c(x)$', 15)
    label(ax, 25, 378, 'BERT fine-tuning', 14, ha='left', color=TEAL, weight='bold')
    block(ax, 25, 402, 226, 91, 'WordPiece tokenizer', 'Pad / truncate: 128 tokens\ninput_ids + attention_mask', fill=PALE_GREEN)
    arrow(ax, [(255, 448), (290, 448)])
    for off in [8, 4]:
        ax.add_patch(Rectangle((300+off, 402-off), 285, 91, facecolor='white', edgecolor=RULE, lw=.6))
    block(ax, 300, 402, 285, 91, 'BERT base encoder', '12 layers · 12 attention heads\nHidden width: 768', fill=PALE_GREEN)
    arrow(ax, [(598, 448), (632, 448)])
    block(ax, 639, 402, 249, 91, 'Classification head', '[CLS] pooling · dropout\nLinear projection: 768 → 5', fill=PALE_GREEN)
    arrow(ax, [(892, 448), (936, 448)])
    label(ax, 1061, 429, '5 class logits', 13.5)
    label(ax, 1061, 460, r'$\hat{y}=\operatorname{arg\,max}_c z_c$', 15)
    label(ax, 25, 524, r'Fine-tuning: 1 epoch · batch size 8 · learning rate $5\times10^{-5}$ · weight decay 0.5. Encoder blocks are schematic.', 12.5, ha='left', color=MUTED)
    save(fig, 'nlp-pipeline')


def comparison(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.45), sharey=True)
    fig.subplots_adjust(left=.23, right=.955, bottom=.27, top=.78, wspace=.31)
    for ax, key, title, xmax in zip(axes, ['accuracy', 'macro_f1'], ['(a) Test accuracy (%)', '(b) Macro F1'], [100, 1]):
        vals = [metrics[m][key] * (100 if key == 'accuracy' else 1) for m in ['svm', 'bert']]
        for y, val, color in zip([0, 1], vals, [BLUE, TEAL]):
            ax.hlines(y, 0, val, color=color, alpha=.2, lw=2)
            ax.scatter(val, y, s=55, color=color, zorder=3)
            ax.annotate(f'{val:.2f}' + ('%' if key == 'accuracy' else ''), (val, y), xytext=(8, 0), textcoords='offset points', va='center', fontsize=12)
        ax.set(xlim=(0, xmax), ylim=(1.6, -.6), yticks=[0, 1], yticklabels=['TF-IDF + LinearSVC', 'BERT · 1 epoch'])
        ax.set_title(title, fontsize=14, pad=13, loc='left')
        ax.set_xticks([0, 25, 50, 75, 100] if xmax == 100 else [0, .25, .5, .75, 1])
        ax.spines[['top', 'left', 'right']].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=13)
        ax.tick_params(axis='x', labelsize=11)
        ax.set_axisbelow(True); ax.grid(axis='x', color='#e7edf1', lw=.7)
    save(fig, 'model-comparison')


def per_class(metrics):
    fig, ax = plt.subplots(figsize=(10, 3.0))
    fig.subplots_adjust(left=.08, right=.98, bottom=.23, top=.78)
    for i in range(5):
        ax.plot([i, i], [metrics['svm']['per_class_f1'][i], metrics['bert']['per_class_f1'][i]], color=RULE, lw=1.2)
    for key, title, color, offset in [('svm', 'TF-IDF + LinearSVC', BLUE, 12), ('bert', 'BERT · 1 epoch', TEAL, -17)]:
        vals = metrics[key]['per_class_f1']
        ax.scatter(range(5), vals, color=color, s=45, label=title, zorder=3)
        for i, val in enumerate(vals):
            other = metrics['bert' if key == 'svm' else 'svm']['per_class_f1'][i]
            offset = 12 if val > other else -17
            ax.annotate(f'{val:.2f}', (i, val), xytext=(0, offset), textcoords='offset points', ha='center', fontsize=11, color=color)
    ax.set(xticks=range(5), xticklabels=metrics['classes'], ylim=(0, 1.03), xlim=(-.5, 4.5), ylabel='F1 score')
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', labelsize=12, length=0, pad=10)
    ax.set_yticks([0, .25, .5, .75, 1]); ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e7edf1', lw=.7)
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.04), ncol=2, fontsize=12)
    save(fig, 'per-class-f1')


def main():
    OUT.mkdir(exist_ok=True)
    evidence = verify_metrics()
    cv_methods(); nlp_method(); comparison(evidence); per_class(evidence)
    print('Verified saved TP4 metrics and generated four research figures.')


if __name__ == '__main__':
    main()
