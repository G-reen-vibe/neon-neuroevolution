"""Verify vectorized dynamics match gymnasium step-for-step."""
import numpy as np
import gymnasium as gym

from neon.envs import ENVS

STATE_SETTER = {
    "Pendulum-v1": lambda ve, gs: ve.s.__setitem__(0, np.asarray(gs)),
}


def compare(env_id, n_steps=300, atol=1e-4):
    cls = ENVS[env_id]
    ge = gym.make(env_id)
    ve = cls(batch=1, rng=np.random.default_rng(0))
    gobs, _ = ge.reset(seed=123)
    ve.reset(seed=999)
    ve.s[0] = np.asarray(ge.unwrapped.state, dtype=np.float64)
    max_err = 0.0
    t = 0
    for t in range(n_steps):
        if cls.discrete:
            a = np.array([t % cls.n_actions])
            go, gr, gterm, gtrunc, _ = ge.step(int(a[0]))
        else:
            a = np.array([[np.sin(t * 0.37) * cls.act_scale]])
            go, gr, gterm, gtrunc, _ = ge.step(a[0].astype(np.float32))
        vo, vr, vd = ve.step(a)
        max_err = max(max_err, float(np.abs(vo[0] - go).max()),
                      float(abs(vr[0] - gr)))
        if gterm or gtrunc:
            break
    assert max_err < atol, f"{env_id}: max_err={max_err}"
    print(f"  {env_id:28s} OK  {t+1:4d} steps  max_err={max_err:.2e}")


if __name__ == "__main__":
    print("Verifying vectorized envs against gymnasium:")
    for env_id in ENVS:
        compare(env_id)
    print("all environments match gymnasium.")
