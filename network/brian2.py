# Refactored from https://brian2.readthedocs.io/en/stable/examples/frompapers.Diehl_Cook_2015.html
import itertools
from collections import defaultdict
from pathlib import Path
from random import randrange
from random import seed as rseed
from struct import unpack

import fire
import matplotlib.pyplot as plt
import numpy as np
from brian2 import (
    Equations,
    Hz,
    Network,
    NeuronGroup,
    PoissonGroup,
    SpikeMonitor,
    Synapses,
    defaultclock,
    ms,
    mV,
    seed,
    volt,
)
from tqdm import tqdm

# Number of training, observation, and testing samples
N_TRAIN = 25_000
N_OBSERVE = 2_000
N_TEST = 1_000

# Random seed value
SEED = 42

# Number of weight save points
N_SAVE_POINTS = 100

# Don't change these values unless you know what you're doing.
N_INP = 784
N_NEURONS = 400
V_EXC_REST = -65 * mV
V_INH_REST = -60 * mV
INTENSITY = 2

# Weights of exc->inh and inh->exc synapses
W_EXC_INH = 10.4
W_INH_EXC = 17.0


def get_run_path(run_name: str) -> Path:
    return Path(__file__).parent.parent / "runs" / run_name


def save_npy(arr, path):
    arr = np.array(arr)
    print("%-9s %-15s => %-30s" % ("Saving", arr.shape, path))
    np.save(path, arr)


def load_npy(path):
    arr = np.load(path)
    print("%-9s %-30s => %-15s" % ("Loading", path, arr.shape))
    return arr


def read_mnist(run_path: Path, training: bool) -> tuple[np.ndarray, np.ndarray]:
    tag = "train" if training else "t10k"
    mnist_path = run_path / "MNIST" / "raw"

    images = open(mnist_path / ("%s-images-idx3-ubyte" % tag), "rb")
    images.read(4)
    n_images = unpack(">I", images.read(4))[0]
    _n_rows = unpack(">I", images.read(4))[0]
    _n_cols = unpack(">I", images.read(4))[0]

    labels = open(mnist_path / ("%s-labels-idx1-ubyte" % tag), "rb")
    labels.read(4)
    x = np.frombuffer(images.read(), dtype=np.uint8)
    x = x.reshape(n_images, -1) / 8.0
    y = np.frombuffer(labels.read(), dtype=np.uint8)
    return x, y


def build_network(data_path: Path, training: bool) -> Network:
    eqs = """
    dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem  : volt (unless refractory)
    i_exc = ge * -v                         : volt
    i_inh = gi * (v_inh_base - v)           : volt
    dge/dt = -ge/(1 * ms)                   : 1
    dgi/dt = -gi/(2 * ms)                   : 1
    dtimer/dt = 1                           : second
    """
    reset = "v = %r; timer = 0 * ms" % V_EXC_REST
    if training:
        exc_eqs = (
            eqs
            + """
        dtheta/dt = -theta / (1e7 * ms)         : volt
        """
        )
        arr_theta = np.ones(N_NEURONS) * 20 * mV
        reset += "; theta += 0.05 * mV"
    else:
        exc_eqs = (
            eqs
            + """
        theta                                   : volt
        """
        )
        arr_theta = load_npy(data_path / "theta.npy") * volt
    exc_eqs = Equations(
        exc_eqs, tau_mem=100 * ms, v_rest=V_EXC_REST, v_inh_base=-100 * mV
    )
    # Note that this neuron has a bit of un unusual refractoriness mechanism:
    # The membrane potential is clamped for 5ms, but spikes are prevented for 50ms
    # This has been taken from the original code.
    ng_exc = NeuronGroup(
        N_NEURONS,
        exc_eqs,
        threshold="v > (theta - 72 * mV) and (timer > 50 * ms)",
        refractory=5 * ms,
        reset=reset,
        method="euler",
        name="exc",
    )
    ng_exc.v = V_EXC_REST
    ng_exc.theta = arr_theta

    inh_eqs = Equations(eqs, tau_mem=10 * ms, v_rest=V_INH_REST, v_inh_base=-85 * mV)
    ng_inh = NeuronGroup(
        N_NEURONS,
        inh_eqs,
        threshold="v > -40 * mV",
        refractory=2 * ms,
        reset="v = -45 * mV",
        method="euler",
        name="inh",
    )
    ng_inh.v = V_INH_REST

    syns_exc_inh = Synapses(ng_exc, ng_inh, on_pre="ge_post += %f" % W_EXC_INH)
    syns_exc_inh.connect(j="i")

    syns_inh_exc = Synapses(ng_inh, ng_exc, on_pre="gi_post += %f" % W_INH_EXC)
    syns_inh_exc.connect("i != j")

    pg_inp = PoissonGroup(N_INP, 0 * Hz, name="inp")

    # During training, inp->exc synapse weights are plastic.
    model = "w : 1"
    on_post = ""
    on_pre = "ge_post += w"
    if training:
        on_pre += "; pre = 1.; w = clip(w - 0.0001 * post1, 0, 1.0)"
        on_post += "post2bef = post2; w = clip(w + 0.01 * pre * post2bef, 0, 1.0); post1 = 1.; post2 = 1."
        model += """
        post2bef                        : 1
        dpre/dt   = -pre/(20 * ms)      : 1 (event-driven)
        dpost1/dt = -post1/(20 * ms)    : 1 (event-driven)
        dpost2/dt = -post2/(40 * ms)    : 1 (event-driven)
        """
        weights = (np.random.random(N_INP * N_NEURONS) + 0.01) * 0.3
    else:
        weights = load_npy(data_path / "weights.npy")

    syns_inp_exc = Synapses(
        pg_inp, ng_exc, model=model, on_pre=on_pre, on_post=on_post, name="inp_exc"
    )
    syns_inp_exc.connect(True)
    syns_inp_exc.delay = "rand() * 10 * ms"
    syns_inp_exc.w = weights

    exc_mon = SpikeMonitor(ng_exc, name="sp_exc")
    net = Network(
        [pg_inp, ng_exc, ng_inh, syns_inp_exc, syns_exc_inh, syns_inh_exc, exc_mon]
    )
    # Initialize
    net.run(0 * ms)
    return net


def show_sample(net, sample, intensity):
    exc_mon = net["sp_exc"]
    prev = exc_mon.count[:]
    net["inp"].rates = sample * intensity * Hz
    net.run(350 * ms)
    # Don't count spikes occuring during the 150 ms rest.
    next = exc_mon.count[:]
    net["inp"].rates = 0 * Hz
    net.run(150 * ms)
    pat = next - prev
    cnt = np.sum(pat)
    if cnt < 5:
        return show_sample(net, sample, intensity + 1)
    return pat


def predict(groups, rates):
    return np.argmax([rates[grp].mean() for grp in groups])


def test(run_name: str) -> None:
    run_path = get_run_path(run_name)
    data_path = run_path / "data"

    conf = np.zeros((10, 10))
    assign = np.load(data_path / "assign.npy")
    groups = [np.where(assign == i)[0] for i in range(10)]

    X, Y = read_mnist(run_path, False)
    net = build_network(data_path, False)
    for i in tqdm(range(N_TEST)):
        ix = randrange(len(X))
        exc = show_sample(net, X[ix], INTENSITY)
        guess = predict(groups, exc)
        real = Y[ix]
        conf[real, guess] += 1

    print("Accuracy: %6.3f" % (np.trace(conf) / np.sum(conf)))
    conf = conf / conf.sum(axis=1)[:, None]
    print(np.around(conf, 2))
    save_npy(conf, data_path / "confusion.npy")


def normalize_plastic_weights(syns):
    conns = np.reshape(syns.w, (N_INP, N_NEURONS))
    col_sums = np.sum(conns, axis=0)
    factors = 78.0 / col_sums
    conns *= factors
    syns.w = conns.reshape(-1)


def stats(net):
    tick = defaultclock.timestep[:]
    cnt = np.sum(net["sp_exc"].count[:])

    inp_exc = net["inp_exc"]
    w_mu = np.mean(inp_exc.w)
    w_std = np.std(inp_exc.w)

    exc = net["exc"]
    theta = exc.theta / mV
    theta_mu = np.mean(theta)
    theta_sig = np.std(theta)
    return [tick, cnt, w_mu, w_std, theta_mu, theta_sig]


def train(run_name: str) -> None:
    run_path = get_run_path(run_name)
    data_path = run_path / "data"

    X, Y = read_mnist(run_path, True)
    n_samples = X.shape[0]
    net = build_network(data_path, True)
    rows = [stats(net) + [-1]]
    w_hist = [np.array(net["inp_exc"].w)]

    ratio = max(N_TRAIN // N_SAVE_POINTS, 1)
    for i in tqdm(range(N_TRAIN)):
        ix = i % n_samples
        normalize_plastic_weights(net["inp_exc"])
        show_sample(net, X[ix], INTENSITY)
        rows.append(stats(net) + [Y[ix]])
        if i % ratio == 0:
            w_hist.append(np.array(net["inp_exc"].w))

    save_npy(rows, data_path / "train_stats.npy")
    save_npy(w_hist, data_path / "train_w_hist.npy")
    save_npy(net["inp_exc"].w, data_path / "weights.npy")
    save_npy(net["exc"].theta, data_path / "theta.npy")


def observe(run_name: str) -> None:
    run_path = get_run_path(run_name)
    data_path = run_path / "data"

    X, Y = read_mnist(run_path, True)
    n_samples = X.shape[0]
    net = build_network(data_path, False)
    rows = [stats(net) + [-1]]
    responses = defaultdict(list)

    for i in tqdm(range(N_OBSERVE)):
        ix = i % n_samples
        sample = X[ix]
        cls = Y[ix]
        exc = show_sample(net, sample, INTENSITY)
        rows.append(stats(net) + [Y[ix]])
        responses[cls].append(exc)

    res = np.zeros((10, N_NEURONS))
    for cls, vals in responses.items():
        res[cls] = np.array(vals).mean(axis=0)

    assign = np.argmax(res, axis=0)
    save_npy(assign, data_path / "assign.npy")
    save_npy(rows, data_path / "observe_stats.npy")


def plot(run_name: str) -> None:
    run_path = get_run_path(run_name)
    data_path = run_path / "data"

    conf = np.load(data_path / "confusion.npy")

    plt.imshow(100 * conf, interpolation="nearest", cmap=plt.cm.Blues)  # type: ignore
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        if conf[i, j] == 0:
            continue
        plt.text(
            j,
            i,
            f"{round(100 * conf[i, j])}%",
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if conf[i, j] > 0.5 else "black",
        )
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


if __name__ == "__main__":
    """
    python -m network.brian2 train --run_name brian
    """
    seed(SEED)
    rseed(SEED)
    fire.Fire({"train": train, "observe": observe, "test": test, "plot": plot})
