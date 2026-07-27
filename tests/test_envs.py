"""Verify vectorized dynamics match gymnasium step-for-step."""
import numpy as np
import gymnasium as gym

from neon.envs import VecCartPole, VecPendulum, VecAcrobot


def compare(env_id, vec_cls, n_steps=200, atol=1e-9):
    rng = np.random.default_rng(0)
    ge = gym.make(env_id)
    ve = vec_cls(batch=1, rng=rng)
    gobs, _ = ge.reset(seed=123)
    vobs = ve.reset(seed=999)
    # align initial state manually (reset RNGs differ)
    if env_id == "CartPole-v1":
        ve.s[0] = ge.unwrapped.state
    elif env_id == "Pendulum-v1":
        ve.th[0], ve.thd[0] = ge.unwrapped.state
    else:
        ve.s[0] = ge.unwrapped.state
    max_err = 0.0
    gdone = False
    for t in range(n_steps):
        if env_id == "Pendulum-v1":
            a = np.array([[np.sin(t * 0.37) * 2.0]])
            go, gr, gterm, gtrunc, _ = ge.step(a[0].astype(np.float32))
        else:
            a = np.array([t % vec_cls.n_actions])
            go, gr, gterm, gtrunc, _ = ge.step(int(a[0]))
        vo, vr, vd = ve.step(a)
        max_err = max(max_err, float(np.abs(vo[0] - go).max()),
                      float(abs(vr[0] - gr)))
        if gterm or gtrunc:
            break
    assert max_err < 1e-4, f"{env_id}: max_err={max_err}"
    print(f"{env_id}: OK over {t+1} steps, max_err={max_err:.2e}")


if __name__ == "__main__":
    compare("CartPole-v1", VecCartPole)
    compare("Pendulum-v1", VecPendulum)
    compare("Acrobot-v1", VecAcrobot)
