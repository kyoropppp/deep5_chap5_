import os 
import numpy as np
import matplotlib.pyplot as plt
from my_em import multivariate_normal, gmm, likelihood

# __file__ = ''

path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
xs  = np.loadtxt(path)
print(xs.shape) # (272, 2)

# params
phis = np.array([0.5, 0.5]) 
mus = np.array([[0.0, 50.0], [0.0, 100.0]])
covs = np.array([np.eye(2), np.eye(2)]) # 2つの2x2単位行列

K = len(phis) # 2
N = len(xs) # 272
MAX_ITERS = 100
THESHOLD = 1e-4     # thresholdのミス？？


# params
phis = np.array([0.5, 0.5]) 
mus = np.array([[0.0, 50.0], [0.0, 100.0]])
covs = np.array([np.eye(2), np.eye(2)]) # 2つの2x2単位行列

mus_history = mus.copy()

K = len(phis) # 2
N = len(xs) # 272
MAX_ITERS = 100
THESHOLD = 1e-4     # thresholdのミス？？

current_likelihood =  likelihood(xs, phis, mus, covs)

for iter in range(MAX_ITERS):
    # E-step
    qs = np.zeros((N, K))   # qs(複数形)
    for n in range(N):
        x = xs[n]
        for k in range(K):
            phi, mu, cov = phis[k], mus[k], covs[k]
            qs[n, k] = phi * multivariate_normal(x, mu, cov)
        qs[n] /= gmm(x, phis, mus, covs)  # qs[n] = qs[n] / gmm(...) 

    # M-step
    qs_sum = qs.sum(axis=0)     # 先に計算しておく
    for k in range(K):
        # 1. phis
        phis[k] = qs_sum[k] / N

        # 2. mus
        c = 0
        for n in range(N):
            c += qs[n, k] * xs[n]
        mus[k] = c / qs_sum[k]

        # 3. covs
        c = 0
        for n in range(N):
            z = xs[n] - mus[k]
            z = z[:, np.newaxis] # 数式に合わせるため列ベクトルへ(形状を(D,1)にする)
            c += qs[n, k] * z @ z.T
            # print(c.shape)
        covs[k] = c / qs_sum[k]

    # 終了判定
    print(f"{current_likelihood:.3f}") # 対数尤度を出力 (単調増加している)

    next_likelihood = likelihood(xs, phis, mus, covs)
    diff = np.abs(next_likelihood - current_likelihood) # 差分の絶対値
    if diff < THESHOLD:
        break
    current_likelihood = next_likelihood

FontSize = 18
plt.rcParams.update({'font.size': FontSize})

# ------ GMMの可視化 ------
def plot_contour(w, mus, covs): #
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

plt.figure(figsize = (10, 8))
plt.scatter(xs[:,0], xs[:,1])
plot_contour(phis, mus, covs)
plt.xlabel('Eruptions(Min)')
plt.ylabel('Waiting(Min)')
plt.show()


# 計算したmu, covに沿って(正規分布から)データを生成
N = 500
new_xs = np.zeros((N,2))
for n in range(N): 
    k = np.random.choice(2, p=phis) # phiの確率分布に基づいて 0 または 1 のいずれかをランダムに選択(= クラスタ)
    mu, cov = mus[k], covs[k]
    new_xs[n] = np.random.multivariate_normal(mu, cov)  # 対応するガウス分布からデータを生成


# GMMの可視化
plt.scatter(xs[:,0], xs[:,1], color = "blue", label = "original")
plt.scatter(new_xs[:,0], new_xs[:,1], color = "orange", label = "generated")
# plot_contour(phis, mus, covs)
plt.xlabel('Eruptions(Min)')
plt.ylabel('Waiting(Min)')
plt.legend()
plt.show()