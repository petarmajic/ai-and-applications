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
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon, Circle
from PIL import Image
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


def image_panel(ax, source, crop, x, y, width, height):
    """Place an exact crop of a saved figure, without changing its pixels."""
    image = Image.open(OUT / source).crop(crop)
    ax.imshow(image, extent=(x, x+width, y+height, y), interpolation='nearest', zorder=3)
    ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=RULE, lw=.6, zorder=4))


def volume(ax, x, y, w, h, depth, color=PALE_GREEN):
    dx, dy = depth, -depth*.45
    ax.add_patch(Polygon([(x,y-h/2),(x+w,y-h/2),(x+w+dx,y-h/2+dy),(x+dx,y-h/2+dy)],
                         facecolor=color, edgecolor=INK, lw=.6))
    ax.add_patch(Polygon([(x+w,y-h/2),(x+w+dx,y-h/2+dy),(x+w+dx,y+h/2+dy),(x+w,y+h/2)],
                         facecolor='#aac8be', edgecolor=INK, lw=.6))
    ax.add_patch(Rectangle((x,y-h/2),w,h,facecolor=color,edgecolor=INK,lw=.7))
    for i in range(1,5):
        ax.plot([x, x+w], [y-h/2+i*h/5]*2, color=TEAL, alpha=.25, lw=.45)
    for i in range(1,4):
        ax.plot([x+i*w/4]*2, [y-h/2,y+h/2],color=TEAL,alpha=.25,lw=.45)


def cv_methods():
    fig, ax = canvas(740)
    label(ax,25,25,'(a) Contrast enhancement: image intensities and their distribution',18,ha='left',weight='bold')
    # The crops remove the old plot titles/axes, retaining the saved image/curve pixels.
    image_panel(ax,'histogram-equalization.png',(10,67,304,263),35,77,190,127)
    image_panel(ax,'histogram-equalization.png',(476,68,1043,262),320,105,190,65)
    image_panel(ax,'histogram-equalization.png',(10,343,304,539),640,77,190,127)
    image_panel(ax,'histogram-equalization.png',(476,344,1043,538),945,105,190,65)
    for x,text in [(130,'Original image'),(415,'Input histogram'),(735,'Enhanced contrast'),(1040,'Output histogram')]:
        label(ax,x,56,text,14)
    arrow(ax,[(232,140),(306,140)])
    arrow(ax,[(520,140),(626,140)],color=BLUE)
    label(ax,575,109,r'$T(I)$',16,color=BLUE)
    arrow(ax,[(840,140),(931,140)])
    label(ax,416,198,'256 intensity bins',12.5,color=MUTED)
    label(ax,1040,198,'Histogram equalization',12.5,color=MUTED)
    ax.plot([25,1175],[233,233],color=RULE,lw=.75)

    label(ax,25,259,'(b) Cell segmentation: foreground mask to separated regions',18,ha='left',weight='bold')
    for source_crop,x,title in [((10,32,390,280),35,'Microscopy input'),
                                ((405,32,785,280),495,'Cleaned binary mask'),
                                ((800,32,1180,280),945,'Watershed boundaries')]:
        image_panel(ax,'cell-segmentation.png',source_crop,x,310,190,124)
        label(ax,x+95,290,title,14)
    arrow(ax,[(234,372),(480,372)])
    label(ax,357,343,'Otsu + opening',15,weight='bold')
    label(ax,357,404,r'$3\times3$ kernel · 2 iterations',12.5,color=MUTED)
    arrow(ax,[(695,372),(930,372)],color=TEAL)
    label(ax,813,337,'Marker-based watershed',14,weight='bold')
    label(ax,813,404,'Distance transform → markers',12.5,color=MUTED)
    label(ax,1040,454,'373 detected regions',14,color=TEAL)
    ax.plot([25,1175],[479,479],color=RULE,lw=.75)

    label(ax,25,506,'(c) Local feature matching: descriptors to correspondences',18,ha='left',weight='bold')
    # Abstract image planes and descriptor rows show representation type only.
    for x,y,symbol in [(49,574,r'$I_A$'),(111,593,r'$I_B$')]:
        ax.add_patch(Rectangle((x,y),86,77,facecolor=PALE_BLUE,edgecolor=INK,lw=.7))
        label(ax,x+43,y+38,symbol,20)
    label(ax,126,688,'Two object views',14)
    arrow(ax,[(214,621),(292,621)])
    for r in range(5):
        for c in range(8):
            ax.add_patch(Rectangle((321+c*14,574+r*17),11,12,
                                   facecolor=PALE_BLUE,edgecolor=RULE,lw=.45))
    label(ax,372,550,'SIFT descriptors',15,weight='bold')
    label(ax,372,688,'Local keypoint representation',12.5,color=MUTED)
    arrow(ax,[(445,621),(521,621)])
    label(ax,678,588,r'2-nearest neighbours',16,weight='bold')
    label(ax,678,626,r'$d_1<0.75\,d_2$',20,color=BLUE)
    label(ax,678,666,'Lowe ratio test',14)
    arrow(ax,[(825,621),(926,621)])
    image_panel(ax,'sift-matching.png',(75,32,478,417),945,540,166,159)
    label(ax,1028,716,'656 retained matches',14)
    label(ax,25,730,'Image panels and histogram traces are exact crops of saved outputs; descriptor glyphs are schematic.',11.5,ha='left',color=MUTED)
    save(fig,'computer-vision-methods')


def nlp_method():
    fig,ax=canvas(655)
    label(ax,25,26,'Medical note classification: sparse features and contextual representations',18,ha='left',weight='bold')
    # One shared cleaned-note input branches into two independently fitted models.
    ax.add_patch(Rectangle((52,123),112,122,facecolor='white',edgecolor=INK,lw=.8))
    for y,w in [(143,77),(158,70),(173,83),(188,62),(203,77),(218,49)]:
        ax.plot([66,66+w],[y,y],color=RULE,lw=2.1)
    label(ax,108,101,'Medical note',15,weight='bold')
    arrow(ax,[(108,252),(108,276)])
    label(ax,108,304,'HTML removal\nLowercase + whitespace\nStop words + stemming',12.5,linespacing=1.5)
    label(ax,108,365,'Shared cleaned text',13,weight='bold')
    arrow(ax,[(198,320),(235,320),(235,175),(284,175)],color=BLUE)
    arrow(ax,[(235,320),(235,450),(284,450)],color=TEAL)

    label(ax,298,77,'TF-IDF + LinearSVC',16,ha='left',color=BLUE,weight='bold')
    label(ax,342,127,'TF-IDF',15,weight='bold')
    label(ax,342,156,'Training-fitted\nvocabulary + IDF',12.5,linespacing=1.4)
    arrow(ax,[(390,175),(427,175)],color=BLUE)
    # A small subset of symbolic positions illustrates sparsity, not measured weights.
    for r in range(11):
        ax.add_patch(Rectangle((439,110+r*12),31,10,edgecolor=RULE,lw=.45,
                               facecolor=BLUE if r in [1,4,8] else 'white'))
    label(ax,456,269,r'$x\in\mathbb{R}^{|V_{\rm train}|}$',16)
    label(ax,456,297,'Sparse features',12.5,color=MUTED)
    arrow(ax,[(478,175),(577,175)],color=BLUE)
    for r in range(5):
        for c in range(9):
            ax.add_patch(Rectangle((592+c*15,133+r*17),13,15,facecolor=PALE_BLUE,edgecolor=RULE,lw=.45))
    label(ax,659,110,'LinearSVC',15,weight='bold')
    label(ax,659,244,r'$f_c(x)=w_c^{\mathsf{T}}x+b_c$',16)
    label(ax,659,272,'5 class decision functions',12.5,color=MUTED)
    arrow(ax,[(736,175),(867,175)],color=BLUE)
    for r in range(5):
        ax.add_patch(Rectangle((879,121+r*23),45,18,facecolor=PALE_BLUE,edgecolor=INK,lw=.6))
    label(ax,903,100,'5 scores',14)
    arrow(ax,[(936,175),(1020,175),(1020,268),(1061,268)],color=BLUE)

    label(ax,298,339,'BERT fine-tuning',16,ha='left',color=TEAL,weight='bold')
    tokens=[r'[CLS]',r'$t_1$',r'$\cdots$',r'[SEP]',r'[PAD]']
    for r,token in enumerate(tokens):
        ax.add_patch(Rectangle((298,385+r*25),78,24,facecolor=PALE_GREEN,edgecolor=RULE,lw=.6))
        label(ax,337,397+r*25,token,12.5)
    label(ax,337,362,'WordPiece',14,weight='bold')
    label(ax,337,538,'128 positions',12.5)
    arrow(ax,[(384,450),(420,450)],color=TEAL)
    label(ax,402,427,'IDs',11.5,color=TEAL)
    volume(ax,431,450,42,116,39)
    label(ax,474,363,'Embeddings',14,weight='bold')
    label(ax,474,538,r'$128\times768$',15)
    arrow(ax,[(519,450),(568,450)],color=TEAL)
    # Repeat glyph plus operator names indicates the standard BERT encoder stack.
    for off in [10,5]:
        ax.add_patch(Rectangle((584+off,370-off),154,165,facecolor='white',edgecolor=RULE,lw=.6))
    ax.add_patch(Rectangle((584,370),154,165,facecolor=PALE_GREEN,edgecolor=INK,lw=.8))
    for y,h,name in [(384,35,'Self-attention'),(425,22,'Add + Norm'),(453,35,'Feed-forward'),(494,22,'Add + Norm')]:
        ax.add_patch(Rectangle((595,y),132,h,facecolor='white',edgecolor=RULE,lw=.5))
        label(ax,661,y+h/2,name,12.5)
    label(ax,661,350,'Encoder × 12',15,weight='bold')
    label(ax,661,558,'12 heads · hidden width 768',12.5,color=MUTED)
    arrow(ax,[(380,506),(397,506),(397,583),(551,583),(551,508),(581,508)],color=TEAL)
    label(ax,477,571,'Attention mask',11.5,color=TEAL)
    arrow(ax,[(755,450),(822,450)],color=TEAL)
    left=[410,430,450,470,490]; right=[400,425,450,475,500]
    for y1 in left:
        for y2 in right:
            ax.plot([843,914],[y1,y2],color='#b8ccc6',lw=.45,zorder=0)
    for x,ys in [(836,left),(921,right)]:
        for y in ys:
            ax.add_patch(Circle((x,y),5,facecolor=PALE_GREEN,edgecolor=INK,lw=.6))
    label(ax,880,366,'[CLS] pooling + head',14,weight='bold')
    label(ax,880,538,'Dropout · linear 768 → 5',12.5)
    arrow(ax,[(935,450),(1020,450),(1020,360),(1061,360)],color=TEAL)

    # Shared label space, separate incoming predictions: no averaging or ensemble.
    ax.add_patch(Rectangle((1068,230),107,170,facecolor='white',edgecolor=RULE,lw=.7))
    for i in range(5):
        ax.add_patch(Circle((1096,251+i*29),4.5,facecolor='white',edgecolor=INK,lw=.7))
        label(ax,1127,251+i*29,rf'$c_{i}$',15)
    label(ax,1121,206,'5 specialties',14,weight='bold')
    label(ax,1121,428,'argmax',15)
    label(ax,1121,455,'Per model',12,color=MUTED)

    # Exact split counts sit in a proportional strip, distinct from architecture.
    ax.add_patch(Rectangle((25,595),200,12,facecolor=PALE_BLUE,edgecolor=RULE,lw=.5))
    ax.add_patch(Rectangle((185,595),40,12,facecolor=TEAL,edgecolor='none'))
    label(ax,25,626,'660 train / 165 held-out',12.5,ha='left')
    label(ax,300,607,'Stratified split · seed 42. Each model is fitted and evaluated separately.',12.5,ha='left',color=MUTED)
    label(ax,300,632,'Representation glyphs are schematic; no feature weights, attention maps or class scores are shown.',11.5,ha='left',color=MUTED)
    save(fig,'nlp-pipeline')


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
