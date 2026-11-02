"""Plot recorded TP4 metrics; does not train or evaluate models."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
metrics = json.loads((ROOT / 'docs/results/tp4_metrics.json').read_text())
OUT = ROOT / 'assets'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none'})
colors = ['#27587a', '#8799aa']
fig, ax = plt.subplots(figsize=(8, 3.3), layout='constrained')
values = [metrics['svm']['accuracy'] * 100, metrics['bert']['accuracy'] * 100]
ax.barh(['TF-IDF + LinearSVC', 'BERT · 1 epoch'], values, color=colors, height=.48)
ax.invert_yaxis()
ax.set(xlim=(0, 100), xlabel='Test accuracy (%)', title='Medical notes classification · 165 held-out notes')
ax.spines[['left', 'bottom']].set_visible(False)
ax.set_axisbelow(True); ax.xaxis.grid(True, color='#e6e9ec')
ax.tick_params(axis='y', length=0)
for i,v in enumerate(values): ax.text(v+1.2, i, f'{v:.2f}%', va='center', weight='bold')
fig.savefig(OUT / 'model-comparison.svg'); plt.close(fig)

classes = metrics['classes']; x = np.arange(len(classes)); width=.34
fig, ax = plt.subplots(figsize=(9, 4.3), layout='constrained')
for offset, key, label, color in [(-width/2,'svm','TF-IDF + LinearSVC',colors[0]),(width/2,'bert','BERT · 1 epoch',colors[1])]:
 vals=metrics[key]['per_class_f1']
 bars=ax.bar(x+offset, vals, width, label=label, color=color)
 ax.bar_label(bars, labels=[f'{v:.2f}' for v in vals], padding=4, fontsize=9)
ax.set(xticks=x, xticklabels=classes, ylim=(0,1.1), ylabel='F1 score', title='Per-class F1 · saved TP4 classification reports')
ax.tick_params(axis='x', labelsize=9)
ax.set_axisbelow(True); ax.yaxis.grid(True,color='#e6e9ec')
ax.legend(frameon=False,loc='upper right',fontsize=9)
fig.savefig(OUT/'per-class-f1.svg');plt.close(fig)
