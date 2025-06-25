import os 
import numpy as np
import matplotlib.pyplot as plt
from my_em import multivariate_normal, gmm, likelihood, plot_contour, plot_cov_ellipse

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

# μ, Σの履歴
mus_history = [mus.copy()]
covs_history = [covs.copy()]

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
    
    # μ, Σの履歴を追加
    mus_history.append(mus.copy())
    covs_history.append(covs.copy())

    next_likelihood = likelihood(xs, phis, mus, covs)
    diff = np.abs(next_likelihood - current_likelihood) # 差分の絶対値
    if diff < THESHOLD:
        break
    current_likelihood = next_likelihood

# --- グラフ設定 --- 
FontSize = 18
plt.rcParams.update({'font.size': FontSize})

# --- 軌跡の準備 --- 
covs_history = np.array(covs_history)   # shape=(iters+1, K, 2, 2)
mus_history = np.array(mus_history)  # shape=(iters+1, K, 2)

# --- μの軌跡を描画 ---

plt.figure(figsize=(10, 8))
plt.scatter(xs[:,0], xs[:,1], alpha=0.5, label="data")
colors = ["tab:red", "tab:blue"]
markers = ["o", "s"]

for k in range(K):
    traj = mus_history[:,k,:]
    plt.plot(traj[:,0], traj[:,1], '-o', label=f"$μ_{k}$ trajectory", color=colors[k])
    plt.scatter(traj[0,0], traj[0,1], marker="x", s=100, color=colors[k], label=f"start $μ_{k}$")
    plt.scatter(traj[-1,0], traj[-1,1], marker="*", s=200, color=colors[k], label=f"final $μ_{k}$")

plt.xlabel('Eruptions (Min)')
plt.ylabel('Waiting (Min)')
plt.title(r'Trajectories of $\mu_k$ over EM iterations')
plt.legend()
plt.grid(True)
plt.show()

# --- Σの軌跡を描画 --- 
ax = plt.gca()
for k in range(K):
    traj = mus_history[:,k,:]
    plt.plot(traj[:,0], traj[:,1], '-o', color=colors[k], label=f"$μ_{k}$ trajectory")
    plt.scatter(traj[0,0], traj[0,1], marker="x", s=100, color=colors[k], label=f"start $μ_{k}$")
    plt.scatter(traj[-1,0], traj[-1,1], marker="*", s=200, color=colors[k], label=f"final $μ_{k}$")
    
    # 各イテレーションごとのΣ_k楕円を描画（10ステップごとに表示）
    step = max(1, len(mus_history)//10)
    for t in range(0, len(mus_history), step):
        plot_cov_ellipse(ax, mus_history[t,k], covs_history[t,k], alpha=0.12, color=colors[k])

    # 最終イテレーションの楕円を濃く
    plot_cov_ellipse(ax, mus_history[-1,k], covs_history[-1,k], alpha=0.7, color=colors[k], lw=2)

plt.xlabel('Eruptions (Min)')
plt.ylabel('Waiting (Min)')
plt.title(r'Trajectory of $\mu_k$ and response of $\Sigma_k$')
plt.legend()
plt.grid(True)
plt.show()

# --- GMM の可視化 --- 
plt.figure(figsize = (10, 8))
plt.scatter(xs[:,0], xs[:,1])
plot_contour(phis, mus, covs, FontSize)
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