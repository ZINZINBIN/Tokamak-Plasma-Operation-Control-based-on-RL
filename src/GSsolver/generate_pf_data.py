import matplotlib.pyplot as plt
from src.GSsolver.util import draw_KSTAR_limiter

fig, ax = plt.subplots(1,1, figsize = (6,8))
ax = draw_KSTAR_limiter(ax)
fig.tight_layout()
plt.savefig("./KSTAR_limiter.png")