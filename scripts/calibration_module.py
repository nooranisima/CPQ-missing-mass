from scripts.utils import *
from scripts.sampling_module import *


def compute_accuracy_and_calibration_threshold_optimal(results_json, alpha = 0.2):

    # Load data from JSON file.
    with open(results_json, "r") as f:
        data = json.load(f)

    total_questions = len(data)
    #print("LENGTH", len(data))
    target_coverage = 1 - alpha

    # To compute the accuracy
    #First get the predicted answer by choosing the one with highest prob
    total_em = 0
    for entry in data:
        true_answer = entry["true_answer"]
        semantic_probs = entry["semantic_probs"]
        best_cid = max(semantic_probs, key=semantic_probs.get)
        if best_cid == "EE":
            predicted = "i am not sure"
        else:
            predicted = entry["clusters"][best_cid]["representative"]
            if isinstance(predicted, dict):
                predicted = predicted.get("text", "")
            if predicted is None:
                predicted = ""

        # Compute the adjusted true answer.
        found_in_cluster = False
        for cid, cluster_data in entry["clusters"].items():
            if cid == "EE":
                continue
            rep = cluster_data.get("representative", "")
            if rep is None:
                continue
            if isinstance(rep, dict):
                rep = rep.get("text", "")
            if rep.strip() == true_answer.strip():
                found_in_cluster = True
                break
        entry["true_label"] = true_answer = true_answer if found_in_cluster else "i am not sure"


        # Compute EM: compare the predicted answer with the true label.
        if best_cid == "EE":
            predicted_text = "i am not sure"
        else:
            predicted_text = entry["clusters"][best_cid]["representative"]
            if isinstance(predicted_text, dict):
                predicted_text = predicted_text.get("text", "")
            if predicted_text is None:
                predicted_text = ""
        em = 1 if predicted_text.strip() == (true_answer if found_in_cluster else "i am not sure").strip() else 0
        total_em += em
    em_accuracy = total_em / total_questions

    # --- Compute scores for each entry in the calibration data ---
    for entry in data:
        true_answer = entry["true_answer"]
        semantic_probs = entry["semantic_probs"]
        true_label = entry["true_label"]
        clusters_info = entry["clusters"]

        # Recompute found_in_cluster for this entry.
        # Compute the adjusted true answer.
        found_in_cluster = False
        id = None
        for cid, cluster_data in entry["clusters"].items():
            if cid == "EE":
                continue
            rep = cluster_data.get("representative", "")
            if rep is None:
                continue
            if isinstance(rep, dict):
                rep = rep.get("text", "")
            if rep.strip() == true_answer.strip():
                found_in_cluster = True
                id = cid
                break

        # Set the score based on whether the true label is found.
        # If found, use 1 - probability of true_label plus some noise.
        if found_in_cluster:
            cluster_prob = semantic_probs.get(id)
            entry["score"] = (1 - cluster_prob) + np.random.uniform(0, 0.001)
            #print("true label", true_label)
        # If the true label is "i am not sure" then assign a score near 1.
        elif true_label == "i am not sure":
            ee_p = semantic_probs.get("EE", 0.0)
            #print("Probability of EE:", ee_p)
            entry["score"] = (2 - ee_p) + np.random.uniform(0,0.001)# 2- probability of EE  + np.random.normal(0, 0.01)
            #print("true label", true_label)

    # find the finite_sample 1-alpha quantile of the scores distribution
    scores = [entry["score"] for entry in data]
    n = len(scores)
    if n == 0:
        raise ValueError("Empty scores list!")

    best_tau = np.quantile(scores, (n+1)/n * (1-alpha))
    # print("best tau", best_tau)

    best_tau_coverage = None

    return em_accuracy, best_tau, best_tau_coverage




def generate_prediction_sets_optimal(input_json, calibration_threshold, output_json, normalize_func=None):

    with open(input_json, "r") as f:
        data = json.load(f)
    print("length of data", len(data))
    #To compute the accuracy
    #First get the predicted answer by choosing the one with highest prob
    total_em = 0
    total_questions = len(data)
    prediction_set_results = []
    for entry in data:
        question = entry["question"]
        true_answer = entry["true_answer"]
        semantic_probs = entry["semantic_probs"]
        best_cid = max(semantic_probs, key=semantic_probs.get)
        if best_cid == "EE":
            predicted = "i am not sure"
        else:
            predicted = entry["clusters"][best_cid]["representative"]
            if isinstance(predicted, dict):
                predicted = predicted.get("text", "")
            if predicted is None:
                predicted = ""

        # Compute the adjusted true answer.
        found_in_cluster = False
        for cid, cluster_data in entry["clusters"].items():
            if cid == "EE":
                continue
            rep = cluster_data.get("representative", "")
            if rep is None:
                continue
            if isinstance(rep, dict):
                rep = rep.get("text", "")
            if rep.strip() == true_answer.strip():
                found_in_cluster = True
                break
        entry["true_label"] = true_answer = true_answer if found_in_cluster else "i am not sure"
        true_label = entry["true_label"]

        # Compute EM: compare the predicted answer with the true label.
        if best_cid == "EE":
            predicted_text = "i am not sure"
        else:
            predicted_text = entry["clusters"][best_cid]["representative"]
            if isinstance(predicted_text, dict):
                predicted_text = predicted_text.get("text", "")
            if predicted_text is None:
                predicted_text = ""
        em = 1 if predicted_text.strip() == (true_answer if found_in_cluster else "i am not sure").strip() else 0
        total_em += em
        ############ Building the prediction set ##############
        #Build the prediction set using the new procedure.
        prediction_set = []
        cumulative_prob = 0.0

        #comptue the score for every cluster ( + noise ) and if its <= threshold add to the prediction set
        ee_p = semantic_probs.get("EE", 0.0)
        score_ee = 2 - ee_p + np.random.uniform(0, 0.001)
        if score_ee <= calibration_threshold:
            prediction_set.append("i am not sure")
        else:
            for cid, cluster_data in entry.get("clusters", {}).items():
            # If the cluster id is "EE", we'll handle it separately (or skip it).
              if cid == "EE":
                  #change here, set the score to 2 - p_ee
                  continue
              else:
                  cluster_score = (1 -  semantic_probs.get(cid)) + np.random.uniform(0, 0.001)
                  if cluster_score <= calibration_threshold:
                      val = entry["clusters"][cid]["representative"]
                      #print ("VAL IS ", val)
                      prediction_set.append(val.get("text",""))

        #print("prediction set", prediction_set)
        prediction_set_results.append({
            "question": question,
            "true_answer": true_label,
            "predicted": predicted,
            "prediction_set": prediction_set,
            "semantic_probs": semantic_probs
        })


        em_accuracy = total_em / total_questions
        #print("TOTAL ACCURACY:", em_accuracy)

    with open(output_json, "w") as f:
        json.dump(prediction_set_results, f, indent=2)

    print(f"Prediction sets saved to {output_json}")



def compute_coverage_and_set_stats(prediction_sets_json, normalize_func=None):
    """
    Loads a JSON file containing prediction sets (as produced by generate_prediction_sets) and computes:

      - Average coverage: the fraction of examples for which the true answer is present in the prediction set.
      - Average prediction set size: the mean number of answers included in the prediction set.

    If a normalization function is provided, it is applied to both the true answer and each prediction in the set
    before checking for a match.

    Parameters:
      prediction_sets_json (str): Path to the JSON file containing prediction set results.
      normalize_func (callable, optional): A function that normalizes a string for comparison.

    Returns:
      coverage (float): Fraction of examples for which the true answer is in the prediction set.
      avg_set_size (float): Average number of answers in the prediction set.
    """
    with open(prediction_sets_json, "r") as f:
        data = json.load(f)

    total = len(data)
    covered = 0
    EE_frac = 0.0
    acc = 0.0
    set_sizes = []

    for entry in data:
        true_ans = entry["true_answer"]
        pred_set = entry["prediction_set"]
        if normalize_func:
            true_ans = normalize_func(true_ans)
            pred_set = [normalize_func(ans) for ans in pred_set]
            #print("True Answer is :", true_ans)
            #print("Prediction Set is :", pred_set)
            #print("\n\n\n\n\n\n")
        # Exclude EE from set size count
        EE = "i am not sure"
        pred_set_no_EE = [ans for ans in pred_set if ans != EE]
        set_sizes.append(len(pred_set_no_EE))
        #set_sizes.append(len(pred_set))

        # Coverage: check if the true answer appears in the prediction set. if EE is in the set, then its covered, else check to see if the true answer is in the set

        if EE in pred_set:
            covered += 1
            EE_frac += 1
        elif true_ans in pred_set:
            covered += 1
            acc += 1

    coverage = covered / total if total > 0 else 0.0
    acc = acc / total if total > 0 else 0.0
    avg_set_size = np.mean(set_sizes) if total > 0 else 0.0

    return coverage, avg_set_size, EE_frac/total, acc