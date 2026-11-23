import random, time
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

#looking at exact determinant
def det_int(M):
    """Exact determinant of an integer matrix."""
    M = [row[:] for row in M]
    n, sign, prev = len(M), 1, 1
    for k in range(n - 1):
        if M[k][k] == 0:
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    M[k], M[i] = M[i], M[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]


def to_matrix(z, o, n, a=0, b=1):
    """Board -> matrix, Z's cells hold a and O's cells hold b."""
    return [[a if (z >> (i * n + j)) & 1 else b for j in range(n)] for i in range(n)]


def singular(z, o, n, a=0, b=1):
    return det_int(to_matrix(z, o, n, a, b)) == 0


def zero_block(z, n):
    """True if the 0s contain an r x c all-zero submatrix with r + c = n + 1."""
    for r in range(1, n + 1):
        for rows in combinations(range(n), r):
            cols = sum(all((z >> (i * n + j)) & 1 for i in rows) for j in range(n))
            if cols >= n + 1 - r:
                return True
    return False


# looking at our solver
def solve(n, z_first=True, z_target=singular):
    """True if Z wins under perfect play. z_target(z, o, n) says if a full board is a Z win.
    Returns (Z wins?, number of positions stored, seconds)."""
    N, full, memo = n * n, (1 << n * n) - 1, {}

    def z_wins(z, o):
        if (z, o) in memo:
            return memo[(z, o)]
        filled = z | o
        if filled == full:
            res = z_target(z, o, n)
        else:
            z_turn = (bin(filled).count("1") % 2 == 0) == z_first
            res = not z_turn                      
            for c in range(N):
                bit = 1 << c
                if filled & bit:
                    continue
                if z_turn and z_wins(z | bit, o):        # emphasis on Z needing to win one
                    res = True
                    break
                if not z_turn and not z_wins(z, o | bit):  
                    res = False
                    break
        memo[(z, o)] = res
        return res

    t = time.time()
    res = z_wins(0, 0)
    return res, len(memo), time.time() - t


def solve_biased(n, pattern):
    """Like solve(), but the order of moves repeats `pattern`, e.g. 'ZOO' = O moves twice."""
    N, full, memo = n * n, (1 << n * n) - 1, {}

    def z_wins(z, o):
        if (z, o) in memo:
            return memo[(z, o)]
        filled = z | o
        if filled == full:
            res = singular(z, o, n)
        else:
            z_turn = pattern[bin(filled).count("1") % len(pattern)] == "Z"
            res = not z_turn
            for c in range(N):
                bit = 1 << c
                if filled & bit:
                    continue
                if z_turn and z_wins(z | bit, o):
                    res = True
                    break
                if not z_turn and not z_wins(z, o | bit):
                    res = False
                    break
        memo[(z, o)] = res
        return res

    return z_wins(0, 0)


# strategy for n>=4
def partner(c, n):
    """Pairs (1,j)-(2,j) and (3,j)-(4,j) (rows 0-3 here); other cells are free (None)."""
    i, j = divmod(c, n)
    return {0: 1, 1: 0, 2: 3, 3: 2}.get(i, None) * n + j if i < 4 else None


def pairing_move(z, o, n, last):
    """The pairing player's move, given the opponent's last cell (or None)."""
    filled = z | o
    empty = [c for c in range(n * n) if not (filled >> c) & 1]
    if last is not None and partner(last, n) is not None and partner(last, n) in empty:
        return partner(last, n)                                   # rule 1: answer in the pair
    free = [c for c in empty if partner(c, n) is None]
    if free:
        return free[0]                                            # rule 2: a free cell
    fresh = [c for c in empty if partner(c, n) in empty]
    return fresh[0] if fresh else empty[0]                        # rule 3: open a new pair


def pairs_mixed(z, o, n):
    """True if every pair holds one 0 and one 1 (so row1+row2 = row3+row4 = all-ones)."""
    return all(((z >> c) & 1) != ((z >> partner(c, n)) & 1) for c in range(2 * n))


def verify_pairing(n, z_first):
    """Play the pairing strategy for Z against EVERY sequence of O moves (memoised).
    Returns True if every reachable final board has all pairs mixed and det = 0."""
    N, full, memo = n * n, (1 << n * n) - 1, {}

    def ok(z, o, last):
        key = (z, o, last)
        if key in memo:
            return memo[key]
        filled = z | o
        if filled == full:
            res = pairs_mixed(z, o, n) and singular(z, o, n)
        elif (bin(filled).count("1") % 2 == 0) == z_first:       
            c = pairing_move(z, o, n, last)
            res = ok(z | 1 << c, o, None)
        else:                                                    
            res = all(ok(z, o | 1 << c, c) for c in range(N) if not (filled >> c) & 1)
        memo[key] = res
        return res

    return ok(0, 0, None), len(memo)


def play(n, z_first, z_policy, o_policy, rng):
    """One game; a policy maps (z, o, n, last, rng) -> cell. Returns the final board."""
    z = o = 0
    last = None
    for k in range(n * n):
        if (k % 2 == 0) == z_first:
            c = z_policy(z, o, n, last, rng); z |= 1 << c
        else:
            c = o_policy(z, o, n, last, rng); o |= 1 << c
        last = c
    return z, o


def random_policy(z, o, n, last, rng):
    filled = z | o
    return rng.choice([c for c in range(n * n) if not (filled >> c) & 1])


def pairing_policy(z, o, n, last, rng):
    return pairing_move(z, o, n, last)


def random_singular_prob(n, z_first, trials, rng):
    """Both players random: the final board is a uniformly random arrangement of
    ceil(n^2/2) (Z first) or floor(n^2/2) (O first) zeros."""
    zeros = (n * n + 1) // 2 if z_first else n * n // 2
    hits = 0
    for _ in range(trials):
        cells = np.ones(n * n, dtype=int)
        order = list(range(n * n)); rng.shuffle(order)
        cells[order[:zeros]] = 0
        hits += np.linalg.matrix_rank(cells.reshape(n, n)) < n
    return hits / trials


# main part
if __name__ == "__main__":
    rng = random.Random(2026)

    print("== Exact solution, original game (Z writes 0 and wants det = 0) ==")
    for n in (1, 2, 3, 4):
        for zf in (True, False):
            res, size, sec = solve(n, zf)
            print(f"n={n}  {'Z first' if zf else 'O first'}:  Z wins = {res}   "
                  f"positions stored = {size:>8}  ({sec:.1f}s, 3^(n^2) = {3**(n*n)})")

    print("\n== Can Z force an r x c block of zeros with r + c = n + 1? ==")
    for n in (2, 3, 4):
        for zf in (True, False):
            res, _, _ = solve(n, zf, lambda z, o, n: zero_block(z, n))
            print(f"n={n}  {'Z first' if zf else 'O first'}:  {res}")

    print("\n== Roles reversed: the 0-player wants det != 0, the 1-player wants det = 0 ==")
    for n in (1, 2, 3, 4):
        for zf in (True, False):
            res, _, _ = solve(n, zf, lambda z, o, n: not singular(z, o, n))
            print(f"n={n}  {'0-player first' if zf else '1-player first'}:  "
                  f"1-player (wants det = 0) wins = {not res}")

    print("\n== Biased game: O writes two 1s for every 0 ==")
    for n in (2, 3, 4):
        for p in ("ZOO", "OZO", "OOZ"):
            print(f"n={n}  order {p}:  Z wins = {solve_biased(n, p)}")

    print("\n== Pairing strategy checked against every possible opponent ==")
    for n in (4, 5):
        for zf in (True, False):
            res, size = verify_pairing(n, zf)
            print(f"n={n}  {'Z first' if zf else 'O first'}:  always singular = {res}"
                  f"   (positions = {size})")

    print("\n== Pairing strategy against random opponents, 2000 games each ==")
    for n in range(4, 13):
        wins = 0
        for g in range(2000):
            z, o = play(n, g % 2 == 0, pairing_policy, random_policy, rng)
            wins += singular(z, o, n)
        print(f"n={n}: Z won {wins}/2000")

    print("\n== Pairing works for any two distinct symbols a != b ==")
    for a, b in [(0, 1), (1, 0), (2, 7), (-3, 5)]:
        ok = 0
        for g in range(500):
            z, o = play(6, g % 2 == 0, pairing_policy, random_policy, rng)
            ok += singular(z, o, 6, a, b)
        print(f"pairing player writes {a}, opponent writes {b}: det = 0 in {ok}/500 games")

    print("\n== Both players random: P(det = 0) ==")
    ns = list(range(1, 13))
    p_z = [random_singular_prob(n, True, 20000, rng) for n in ns]
    p_o = [random_singular_prob(n, False, 20000, rng) for n in ns]
    for n, a, b in zip(ns, p_z, p_o):
        print(f"n={n:2d}  Z first {a:.4f}   O first {b:.4f}")

    # determinant histogram for random 4 x 4 games 
    dets = []
    for _ in range(20000):
        cells = [1] * 16
        order = list(range(16)); rng.shuffle(order)
        for c in order[:8]:
            cells[c] = 0
        dets.append(det_int([cells[4 * i:4 * i + 4] for i in range(4)]))
    dets = np.array(dets)
    print(f"\nrandom 4x4 games: P(det = 0) = {np.mean(dets == 0):.3f}, "
          f"det values {sorted(set(dets.tolist()))}")

    # all the figures i used
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    vals, counts = np.unique(dets, return_counts=True)
    ax[0].bar(vals, counts / len(dets), color=["tab:red" if v == 0 else "tab:blue" for v in vals])
    ax[0].set_xlabel("determinant"); ax[0].set_ylabel("proportion of games")
    ax[0].set_title(r"(a) $n=4$, both players random")
    ax[1].plot(ns, p_z, "o-", label="Z first")
    ax[1].plot(ns, p_o, "s--", label="O first")
    ax[1].plot(ns[3:], [1] * len(ns[3:]), "-", color="tab:green", lw=2, label="Z uses pairing ($n\\geq4$)")
    ax[1].set_xlabel("$n$"); ax[1].set_ylabel("P(det = 0)")
    ax[1].set_title("(b) chance that Z wins")
    ax[1].set_xticks(ns); ax[1].legend()
    plt.tight_layout(); plt.savefig("fig_random.png", dpi=200); plt.close()

    # one game of pairing vs random on a 7 x 7 board
    n = 7
    z, o = play(n, False, pairing_policy, random_policy, random.Random(7))
    M = np.array(to_matrix(z, o, n))
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(M, cmap="Greys", vmin=-0.4, vmax=1.4)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, M[i, j], ha="center", va="center",
                    color="white" if M[i, j] else "black", fontsize=12)
    for top in (0, 2):
        for j in range(n):
            ax.add_patch(plt.Rectangle((j - 0.45, top - 0.45), 0.9, 1.9, fill=False,
                                       ec="tab:red", lw=1.6))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"det = {det_int(M.tolist())}")
    plt.tight_layout(); plt.savefig("fig_pairing.png", dpi=200); plt.close()
    print("\nfigures saved: fig_random.png, fig_pairing.png")
