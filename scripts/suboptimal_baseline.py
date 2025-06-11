from scripts.utils import *
def compute_accuracy_and_calibration_threshold(results_json, alpha=0.2):
    #impelments the suboptimal, yet valid calibration that is used as baseline 
    # Load data from JSON file.

    with open(results_json, "r") as f:
        data = json.load(f)
    #data = data['simulation_outputs']
    total_questions = len(data)
    print("LENGTH", len(data))
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
        #true_answer = true_answer if found_in_cluster else "i am not sure"
        #print("true answer", true_answer)




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

    # Now try candidate thresholds.
    best_tau = 0
    best_coverage_diff = float('inf')
    best_tau_coverage = None
    candidate_results = []  # list of (tau, coverage, coverage_diff)

    # c corresponds to 1 - tau; candidate values in [0, 1] (100 values)
    candidate_cs = np.linspace(0, 1, 5000)
    for c in candidate_cs:

        tau = 1 - c
        covered = 0

        for entry in data:
            semantic_probs = entry["semantic_probs"]
            true_label = entry["true_label"]
            clusters_info = entry["clusters"]

            #Build the prediction set using the new procedure.
            prediction_set = []
            cumulative_prob = 0.0

            # Step 1: Check the "EE" cluster.
            ee_prob = semantic_probs.get("EE", 0.0)
            if ee_prob >= tau:
                prediction_set.append("i am not sure")
                cumulative_prob += ee_prob
            #else:
            # Step 2: Sort non-"EE" clusters in descending order.
            non_ee_clusters = [(cid, prob) for cid, prob in semantic_probs.items() if cid != "EE"]
            non_ee_clusters.sort(key=lambda x: x[1], reverse=True)

            # Step 3: Add clusters until cumulative probability reaches c_value.
            for cid, prob in non_ee_clusters:
                if cumulative_prob >= c:
                    break
                rep = clusters_info[cid]["representative"]

                if isinstance(rep, dict):
                    rep = rep.get("text", "")
                prediction_set.append(rep)
                cumulative_prob += prob

            #print("For Tau = ", tau, " prediction set is", prediction_set)


            # Check if the true cluster is in the prediction set.
            EE = "i am not sure"
            if EE in prediction_set:
                covered += 1
            elif true_label in prediction_set:
                covered += 1


        coverage = covered / total_questions
        if coverage < target_coverage:
            continue
        else:
            best_tau = tau
            #print("final tau", best_tau)
            break
        # if best_tau == None:
        #     best_tau = tau
    return em_accuracy, best_tau, best_tau_coverage



def generate_prediction_sets(input_json, calibration_threshold, output_json, normalize_func=None):
    
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

        # Step 1: Check the "EE" cluster.
        ee_prob = semantic_probs.get("EE", 0.0)
        if ee_prob >= calibration_threshold:
            prediction_set.append("i am not sure")
            cumulative_prob += ee_prob

        # Step 2: Sort non-"EE" clusters in descending order.
        non_ee_clusters = [(cid, prob) for cid, prob in semantic_probs.items() if cid != "EE"]
        non_ee_clusters.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Add clusters until cumulative probability reaches 1 - calib_thres.
        for cid, prob in non_ee_clusters:
            if cumulative_prob >= 1 - calibration_threshold:
                break
            rep = entry["clusters"][cid]["representative"]

            if isinstance(rep, dict):
                rep = rep.get("text", "")
            prediction_set.append(rep)
            cumulative_prob += prob


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