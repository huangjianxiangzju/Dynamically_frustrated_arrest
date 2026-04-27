"""
rebuild_contacts_fast.py
Two-step approach:
  Step 1: extract heavy atom coordinates → save as .npy (fast, once)
  Step 2: workers read only their frame slice from .npy (very fast, low memory)
"""

import os
# ======================================================================
# CRITICAL 24-CORE OPTIMIZATION:
# Prevent NumPy/SciPy from multithreading inside the multiprocessing workers.
# This stops 24 workers from spawning 576 total threads and freezing the CPU.
# Must be set BEFORE importing numpy or scipy.
# ======================================================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import pickle, re, time
import networkx as nx
import mdtraj as md
from multiprocessing import Pool

# ======================================================================
# USER SETTINGS
# ======================================================================
STATES = {
    '6nt':  ('6nt/traj_rep1_5.xtc',   '6nt/ref_protein.pdb',   '6nt/gccm_full.dat'),
    '8nt':  ('8nt/traj_rep1_5.xtc',   '8nt/ref_protein.pdb',   '8nt/gccm_full.dat'),
    '10nt': ('10nt/traj_rep1_5.xtc',  '10nt/ref_protein.pdb',  '10nt/gccm_full.dat'),
    '12nt': ('12nt/traj_rep1_5.xtc',  '12nt/ref_protein.pdb',  '12nt/gccm_full.dat'),
    '14nt': ('14nt/traj_rep1_5.xtc',  '14nt/ref_protein.pdb',  '14nt/gccm_full.dat'),
    '16nt': ('16nt/traj_rep1_5.xtc',  '16nt/ref_protein.pdb',  '16nt/gccm_full.dat'),
    '18nt': ('18nt/traj_rep1_5.xtc',  '18nt/ref_protein.pdb',  '18nt/gccm_full.dat'),
}

DISTANCE_CUTOFF = 4.5    # Å
FRAME_CUTOFF    = 0.75
MAX_RESIDUES    = 1368
STRIDE          = 1     # your current setting

# Set to match your exact hardware (24 cores)
N_WORKERS       = 24     

OUT_PKL_SUFFIX  = 'network_G_nierzwicki.pkl'
NPY_SUFFIX      = 'heavy_coords1.npy'   # intermediate file
# ======================================================================


def load_gc_matrix(gccm_file, n_res):
    with open(gccm_file, 'r') as f:
        content = f.read()
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', content)
    N = int(np.round(np.sqrt(len(numbers))))
    full = np.array([float(x) for x in numbers[-N*N:]]).reshape(N, N)
    gc = np.abs(full[:n_res, :n_res])
    np.fill_diagonal(gc, 1.0)
    return gc


# ── Step 1: extract coords to npy ────────────────────────────────────

def extract_coords_to_npy(traj_file, top_file, npy_file, label):
    """
    Load trajectory once, extract heavy atom coords, save as .npy.
    Shape: (n_frames, n_heavy_atoms, 3)  dtype: float32
    This is the only time the full trajectory is loaded.
    After this, workers read slices directly — no MDTraj needed.
    """
    if os.path.isfile(npy_file):
        # load metadata only
        coords = np.load(npy_file, mmap_mode='r')
        print(f'  [{label}] NPY already exists: '
              f'{coords.shape}  (skipping extraction)')
        return coords.shape[0], coords.shape[1]

    print(f'  [{label}] Extracting heavy atom coords → {npy_file}')
    t0 = time.time()

    traj = md.load(traj_file, top=top_file, stride=STRIDE)
    heavy_idx = traj.topology.select(
        f'protein and element != H and resid 0 to {MAX_RESIDUES-1}')
    traj = traj.atom_slice(heavy_idx)

    # coords in Å (MDTraj uses nm, convert to Å for distance calc)
    coords = (traj.xyz * 10.0).astype(np.float32)

    np.save(npy_file, coords)
    print(f'  [{label}] Saved {coords.shape} → {npy_file}  '
          f'({os.path.getsize(npy_file)/1e9:.2f} GB)  '
          f'[{time.time()-t0:.1f}s]')

    # also save residue mapping
    res_file = npy_file.replace('.npy', '_resmap.npy')
    atom_res = np.array([
        traj.topology.atom(i).residue.index
        for i in range(traj.topology.n_atoms)], dtype=np.int32)
    np.save(res_file, atom_res)

    return coords.shape[0], coords.shape[1]


# ── Step 2: worker reads npy slice ───────────────────────────────────

def _worker_npy(args):
    """
    Worker reads only its frame slice from .npy via memory-map.
    Memory per worker = (end-start) frames × n_atoms × 3 × 4 bytes
    """
    npy_file, res_file, start_frame, end_frame, \
        cutoff, n_res = args

    coords_all = np.load(npy_file, mmap_mode='r')
    coords = np.array(coords_all[start_frame:end_frame])  # (nf, natom, 3)
    atom_res = np.load(res_file)  # (natom,)

    n_frames = coords.shape[0]
    n_atoms  = coords.shape[1]

    contact_count = np.zeros((n_res, n_res), dtype=np.float32)

    for fi in range(n_frames):
        pos = coords[fi]  # (n_atoms, 3) in Å

        ATOM_CHUNK = 500
        for ai in range(0, n_atoms, ATOM_CHUNK):
            ai_end = min(ai + ATOM_CHUNK, n_atoms)
            pos_i  = pos[ai:ai_end]          # (chunk, 3)
            ri     = atom_res[ai:ai_end]     # (chunk,)

            diff = pos_i[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.sqrt((diff**2).sum(axis=2))
            in_contact = dist < cutoff  # (chunk, n_atoms)

            for local_i, global_ai in enumerate(range(ai, ai_end)):
                resi = ri[local_i]
                if resi >= n_res:
                    continue
                contact_atoms = np.where(in_contact[local_i])[0]
                for aj in contact_atoms:
                    resj = atom_res[aj]
                    if resj >= n_res or resj == resi:
                        continue
                    contact_count[resi, resj] += 1
                    contact_count[resj, resi] += 1

            del diff, dist, in_contact

    contact_count = np.minimum(contact_count, n_frames)
    return contact_count, n_frames


def _worker_npy_fast(args):
    """
    Faster worker using scipy KDTree for neighbor search.
    Much faster than brute-force distance matrix.
    """
    npy_file, res_file, start_frame, end_frame, \
        cutoff, n_res = args

    from scipy.spatial import cKDTree

    coords_all = np.load(npy_file, mmap_mode='r')
    coords     = np.array(coords_all[start_frame:end_frame])
    atom_res   = np.load(res_file)

    n_frames = coords.shape[0]
    contact_count = np.zeros((n_res, n_res), dtype=np.float32)

    for fi in range(n_frames):
        pos  = coords[fi]           # (n_atoms, 3)
        tree = cKDTree(pos)

        # find all pairs within cutoff
        pairs = tree.query_pairs(cutoff, output_type='ndarray')

        if len(pairs) > 0:
            ri = atom_res[pairs[:, 0]]
            rj = atom_res[pairs[:, 1]]
            mask = (ri != rj) & (ri < n_res) & (rj < n_res)
            ri, rj = ri[mask], rj[mask]
            np.add.at(contact_count, (ri, rj), 1)
            np.add.at(contact_count, (rj, ri), 1)

    contact_count = np.minimum(contact_count, n_frames)

    return contact_count, n_frames


def compute_contact_freq(npy_file, res_file, n_frames,
                          n_res, label):
    """Parallel contact computation from pre-extracted npy."""

    frames_per_w = max(1, n_frames // N_WORKERS)
    jobs = []
    for w in range(N_WORKERS):
        s = w * frames_per_w
        e = (w+1)*frames_per_w if w < N_WORKERS-1 else n_frames
        if s < n_frames:
            jobs.append((npy_file, res_file, s, e,
                         DISTANCE_CUTOFF, n_res))

    n_jobs = len(jobs)
    mem_per_w = frames_per_w * 11172 * 3 * 4 / 1e9
    
    # Check total memory vs 62GB limitation: 
    # Even heavily loaded, 24 workers * memory_per_worker will easily fit into your 62GB
    print(f'  [{label}] {n_jobs} workers × '
          f'~{frames_per_w} frames, '
          f'~{mem_per_w:.2f} GB/worker')

    contact_sum  = np.zeros((n_res, n_res), dtype=np.float32)
    total_frames = 0
    completed    = 0
    t0 = time.time()

    with Pool(processes=n_jobs) as pool:
        for cnt, nf in pool.imap_unordered(_worker_npy_fast, jobs):
            contact_sum  += cnt
            total_frames += nf
            completed    += 1
            elapsed = time.time() - t0
            eta     = elapsed/completed*(n_jobs-completed)
            print(f'\r  [{label}] {completed}/{n_jobs} done  '
                  f'{elapsed:.0f}s  ~{eta:.0f}s remaining    ',
                  end='', flush=True)
    print()

    contact_freq = contact_sum / total_frames
    n_contacts   = int(np.sum(contact_freq >= FRAME_CUTOFF) // 2)
    print(f'  [{label}] Contacts '
          f'(>={FRAME_CUTOFF*100:.0f}%): {n_contacts}')
    return contact_freq


def build_network(gc, contact_matrix, residue_ids):
    n_res = len(residue_ids)
    G = nx.Graph()
    G.add_nodes_from(range(n_res))
    eps = 1e-6
    n_edges = 0
    for i in range(n_res):
        for j in range(i+1, n_res):
            if (contact_matrix[i,j] >= FRAME_CUTOFF
                    and gc[i,j] > 0):
                w = float(-np.log(gc[i,j] + eps))
                G.add_edge(i, j,
                           weight=w,
                           gc=float(gc[i,j]),
                           contact_freq=float(contact_matrix[i,j]))
                n_edges += 1
    print(f'  Network: {G.number_of_nodes()} nodes, {n_edges} edges')
    return G


def get_residue_ids(traj_file, top_file):
    traj = md.load(traj_file, top=top_file, stride=999999)
    heavy_idx = traj.topology.select(
        f'protein and element != H and resid 0 to {MAX_RESIDUES-1}')
    traj = traj.atom_slice(heavy_idx)
    return [r.resSeq for r in traj.topology.residues][
            :min(traj.topology.n_residues, MAX_RESIDUES)]


def process_state(label):
    traj, top, gccm = STATES[label]
    state_dir = os.path.dirname(traj)
    npy_file  = os.path.join(state_dir, NPY_SUFFIX)
    res_file  = npy_file.replace('.npy', '_resmap.npy')
    out_pkl   = os.path.join(state_dir, OUT_PKL_SUFFIX)

    print(f'\n{"="*55}')
    print(f'  [{label}]')
    print(f'{"="*55}')
    t_start = time.time()

    # Step 1: extract coords to npy (once)
    n_frames, n_atoms = extract_coords_to_npy(
        traj, top, npy_file, label)

    # get residue IDs and n_res
    residue_ids = get_residue_ids(traj, top)
    n_res = len(residue_ids)
    print(f'  [{label}] n_frames={n_frames}, '
          f'n_atoms={n_atoms}, n_res={n_res}')

    # Step 2: parallel contact computation
    contact_freq = compute_contact_freq(
        npy_file, res_file, n_frames, n_res, label)

    # Step 3: GC matrix
    gc = load_gc_matrix(gccm, n_res)

    # Step 4: build network
    G = build_network(gc, contact_freq, residue_ids)

    # Step 5: save
    with open(out_pkl, 'wb') as f:
        pickle.dump((G, residue_ids), f)
    print(f'  [SAVED] {out_pkl}')
    print(f'  [{label}] Total: {(time.time()-t_start)/60:.1f} min')
    return label, G.number_of_edges()


def main():
    print('='*55)
    print('  Contact network — Nierzwicki 2021 standard')
    print(f'  Heavy atom < {DISTANCE_CUTOFF}Å, '
          f'>={FRAME_CUTOFF*100:.0f}% frames')
    print(f'  Stride={STRIDE}  Workers={N_WORKERS}')
    print(f'  Method: npy cache + scipy KDTree (24-Core Tuned)')
    print('='*55)
    print('\nStarting in 3s... (Ctrl+C to cancel)')
    time.sleep(3)

    results = []
    for label in STATES:
        lbl, n_edges = process_state(label)
        results.append((lbl, n_edges))

    print('\n' + '='*55)
    print('Done. Summary:')
    for lbl, n_edges in results:
        print(f'  {lbl}: {n_edges} edges')
    print(f'\nOutput: <state_dir>/{OUT_PKL_SUFFIX}')
    print(f'Next: PKL_NAME = "{OUT_PKL_SUFFIX}"')
    print('='*55)


if __name__ == '__main__':
    # Ensure Windows/Linux cross-compatibility for multiprocess spawning
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
