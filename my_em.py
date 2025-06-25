import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 正規分布 ( 書籍では(3.1)式 )
def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    d = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** d * det)
    y = z * np.exp((x - mu).T @ inv @ (x-mu) / -2.0)
    return y

# 混合ガウスモデル( 書籍では(4.3)式 )
def gmm(x, phis, mus, covs):
    K = len(phis)
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y

# 対数尤度を求める関数
def likelihood(xs, phis, mus, covs):
    eps = 1e-8 # log(0)を防ぐための微小値
    L = 0
    N = len(xs)
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y + eps)
    return L / N


# ------ GMMの可視化 ------
def plot_contour(w, mus, covs, FontSize = 18): #
    x = np.arange(1, 6, 0.1)
    y = np.arange(40, 100, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            for k in range(len(mus)):
                mu, cov = mus[k], covs[k]
                Z[i, j] += w[k] * multivariate_normal(x, mu, cov)

    cs = plt.contour(X, Y, Z) # 等高線の情報を取得
    plt.clabel(cs, inline=True, fontsize=FontSize)  # 等高線を表示


# ----- Σの描画 ----- 
def plot_cov_ellipse(ax, mu, cov, n_std=2.0, **kwargs):
    """
    共分散行列を楕円で描画
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellip = Ellipse(xy=mu, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellip)