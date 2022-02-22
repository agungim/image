import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from numpy.fft import rfft, rfftfreq
from numpy.fft import fft, fftshift
from scipy.signal import blackman
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt

filename="image/1k.txt"


def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line + '\n' + content)
        #f.write(line.rstrip('\r\n') + '\n' + content)
line_prepender(filename, 'Value,number')
df = pd.read_csv(filename) 
print(df)


#sel_df = get_indiv_emission(_tmp_df, 'Natural Gas Electric Power Sector CO2 Emissions')

sel_df=df.copy()

from scipy.signal import blackman

def plot_ori_window(time_: np.ndarray, 
                    val_orig: pd.core.series.Series, 
                    val_window: pd.core.series.Series):
    plt.figure(figsize=(14, 10))
    plt.plot(time_, val_orig, label='raw')
    plt.plot(time_, val_window, label='windowed time')
    plt.legend()
    plt.show()
    return


def plot_ft_result(val_orig_psd: np.ndarray, 
                   val_widw_psd: np.ndarray,
                   ft_smpl_freq: np.ndarray,
                   pos: int=2, annot_mode: bool=True
                  ):
    """
    For PSD graph, the first few points are removed because it represents the baseline (or mean)
    """
    plt.figure(figsize=(14, 10))
    plt.plot(ft_smpl_freq[pos: ], val_orig_psd[pos: ], label='psd original value')
    plt.plot(ft_smpl_freq[pos: ], val_widw_psd[pos: ], label='psd windowed value')
    if annot_mode:
        annot_max(ft_smpl_freq[pos:], abs(val_widw_psd[pos: ]))
        
    plt.xlabel('frequency (1/Year)')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    return

def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax),  xytext=(0.94, 0.96), **kw)

_val_orig = sel_df['Value']
_time = np.linspace(0, len(_val_orig) / 12, len(_val_orig), endpoint=False)
_val_widw = (_val_orig - np.median(_val_orig)) * blackman(len(_val_orig))

# rfft() method is used because the data is real valued
# rfftfreq() method is used to generate the frequency list (usage with rfft)
#  remark: the array f contains the frequency bin centers in cycles per unit of the sample spacing
#    with zero at the start. If the sample spacing is in years, then the frequency unit is cycles/year.
_val_orig_psd = abs(rfft(_val_orig))
_val_widw_psd = abs(rfft(_val_widw))
_val_freqs = rfftfreq(len(_val_orig), d=1./12.)

plot_ori_window(time_=_time, val_orig=_val_orig, val_window=_val_widw)
plot_ft_result(val_orig_psd=_val_orig_psd, val_widw_psd=_val_widw_psd, ft_smpl_freq=_val_freqs)
