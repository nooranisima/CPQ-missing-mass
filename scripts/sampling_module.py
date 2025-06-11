
import math
from scripts.utils import *
from collections import Counter
import random 

def simulate_and_cluster(
    prediction_history,
    saved_clusters,
    sampling_criteria: str,
    fixed_queries: int = None,
    derivative_type: str = "raw",
    lambda_threshold: float = 0.05,
    epsilon_threshold: float = 0.05,
    h: int = 20,
    smoothing_factor: float = 0.75,
    min_queries: int = 2,
    max_queries: int = 50,
    initial_derivative: float = 0.05
):
    """
    Sampling & clustering:
    - baseline: sample exactly `fixed_queries` times
    - derivative: stop when derivative ≤ lambda_threshold (after min_queries)
        derivative_type: `raw`, `MA`, or `bootstrap`
        before min_queries, derivative is set to `initial_derivative`
    Returns:
      responses, clusters, simulated_history
    """
    # Precompute normalized saved responses
    saved_responses_map = {}
    for orig_cid, cluster in saved_clusters.items():
        if orig_cid == "EE":
            continue
        if "responses" in cluster and cluster["responses"]:
            texts = [
                normalize_answer(item["text"]) if isinstance(item, dict)
                else normalize_answer(item)
                for item in cluster["responses"]
            ]
        else:
            rep = cluster.get("representative", "")
            texts = [
                normalize_answer(rep["text"]) if isinstance(rep, dict)
                else normalize_answer(rep)
            ]
        saved_responses_map[orig_cid] = texts

    clusters = {}
    responses = []
    resp_to_sim = {}
    freq_counts = {}
    orig_to_sim = {}
    next_new_cluster_id = 0

    simulated_history = []
    T = 0
    P_EE = 1.0
    prev_derivative = initial_derivative
    smoothed_N2 = 0.0
    for entry in prediction_history[:max_queries]:
        T += 1
        text = entry["response"]
        norm_text = normalize_answer(text)

        # assign to simulation cluster
        matched = next((oc for oc, txts in saved_responses_map.items()
                        if norm_text in txts), None)
        if matched and matched in orig_to_sim:
            sim_cid = orig_to_sim[matched]
        elif matched:
            sim_cid = str(next_new_cluster_id)
            orig_to_sim[matched] = sim_cid
            next_new_cluster_id += 1
        else:
            sim_cid = f"new_{next_new_cluster_id}"
            next_new_cluster_id += 1

        # record response
        responses.append(text)
        resp_to_sim[text] = sim_cid
        clusters.setdefault(sim_cid, {"responses": []})["responses"].append({
            "text": text,
            "log_prob": entry["log_prob"]
        })

        # update frequency counts
        freq = len(clusters[sim_cid]["responses"])
        if freq > 1:
            freq_counts[freq - 1] = freq_counts.get(freq - 1, 0) - 1
        freq_counts[freq] = freq_counts.get(freq, 0) + 1

        # compute missing mass
        P_EE = simple_good_turing(freq_counts)
        N1 = freq_counts.get(1,0)
        N2 = freq_counts.get(2,0)

        # compute derivative if needed
        deriv = None
        if sampling_criteria == "derivative":
            if T <= min_queries:
                s0 = min_queries/2
                deriv = 2* s0 / (min_queries**2)
                prev_derivative = deriv
            else:
                N2 = freq_counts.get(2,0)
                raw = estimate_derivative_missing_mass(freq_counts)
                if derivative_type == "raw":
                    deriv = raw
                    #print("raw_derivative", raw)
                elif derivative_type == "MA":
                    smoothed_N2 = smoothing_factor * smoothed_N2 + (1- smoothing_factor) * N2
                    deriv = 2 * smoothed_N2 / (T**2)

                elif derivative_type == "bootstrap":
                    boots = []
                    for _ in range(h):
                        sample_resp = random.choices(responses, k=T)
                        counts = Counter(resp_to_sim[r] for r in sample_resp)
                        fc = {}
                        for c in counts.values():
                            fc[c] = fc.get(c, 0) + 1
                        boots.append(estimate_derivative_missing_mass(fc))
                    deriv = sum(boots) / h
                    #print("T : ", T, "Bootstrap Derivative: ", deriv)
                prev_derivative = deriv


        # prepare record
        record = {
            "response": text,
            "log_prob": entry["log_prob"],
            "P_EE": P_EE,
            "derivative": deriv,
            "stop_idx": None,
            "N1": N1,
            "N2": N2
        }

        # stopping criteria
        stop = False
        if sampling_criteria == "baseline":
            if fixed_queries and T >= fixed_queries:
                stop = True
        elif sampling_criteria == "missing_mass":
            if T >= min_queries and P_EE <= epsilon_threshold:
                stop = True
        elif sampling_criteria == "derivative":
            if T >= min_queries and deriv is not None and deriv <= lambda_threshold:
                stop = True

        if stop:
            record["stop_idx"] = T
            simulated_history.append(record)
            break

        simulated_history.append(record)

    ##############this is using good turing for other seen cluster probabilities################
    # Step 1: Count how many clusters have each frequency
    cluster_sizes = [len(data["responses"]) for cid, data in clusters.items()]
    r_counts = Counter(cluster_sizes)

    # Step 2: Estimate Good–Turing adjusted frequencies
    adjusted_counts = {}
    for r in r_counts:
        r_plus_1 = r + 1
        if r_plus_1 in r_counts:
            adjusted = (r + 1) * r_counts[r_plus_1] / r_counts[r]
        else:
            adjusted = r  # fallback: no correction if r+1 not observed
        adjusted_counts[r] = adjusted

    # Step 3: Assign adjusted counts to each cluster
    observed_probs = {}
    for cid, data in clusters.items():
        r = len(data["responses"])
        adjusted_count = adjusted_counts.get(r, r)
        observed_probs[cid] = adjusted_count
    ############# END ##############

    total_obs = sum(observed_probs.values())
    semantic_probs = {
        cid: (p / total_obs) * (1 - P_EE) if total_obs > 0 else 0.0
        for cid, p in observed_probs.items()
    }
    semantic_probs["EE"] = P_EE
    semantic_entropy = -sum(
        p * math.log(p) for p in semantic_probs.values() if p > 0
    )
    # Attach the computed probability to each simulated cluster.
    for cid in clusters:
        clusters[cid]["probability"] = semantic_probs[cid]
    clusters["EE"] = {"responses": [], "probability": P_EE}

    return responses, clusters, semantic_probs, simulated_history, T