#!/usr/bin/env python3
"""
Evaluation script: run 5 keyboard-controlled games and 5 hand-tracking games,
then log accuracy (percent won) and other metrics per user. Results are appended
to evaluation_results.txt and evaluation_results.json for analysis.
"""

import json
import os
import sys
import time
import subprocess
from datetime import datetime
 
RESULTS_TXT = "evaluation_results.txt"
RESULTS_JSON = "evaluation_results.json"
GAMES_PER_MODE = 5
 
# --- Subprocess entry point ---
# When this script is called with arguments, it runs a single game and exits.
# The parent process reads the result from the subprocess exit code:
#   0 = win, 1 = loss, 2 = quit
if len(sys.argv) == 3 and sys.argv[1] == "--run-game":
    control_type = sys.argv[2]
    from claw_game import run_one_game
    outcome = run_one_game(control_type=control_type)
    if outcome is True:
        sys.exit(0)   # win
    elif outcome is False:
        sys.exit(1)   # loss
    else:
        sys.exit(2)   # quit
 
 
def run_one_game_subprocess(control_type):
    """
    [Short Description]: Spawns a fresh subprocess to run a single claw game and returns the outcome.
    [AI Declaration]: Generated using Claude with the prompt: "fix computer freezing after running evaluate_claw_modes.py"
    Args:
        control_type (str): The control mode for the game, either "hand" or "keyboard".
    Returns:
        bool or None: True if the player won, False if the player lost, None if the player quit or the subprocess crashed.
    """
    result = subprocess.run(
        [sys.executable, __file__, "--run-game", control_type],
        # Inherit stdio so the game window and prints work normally
    )
    code = result.returncode
    if code == 0:
        return True
    elif code == 1:
        return False
    else:
        return None
    
def run_evaluation_round(control_type, num_games):
    """
    [Short Description]: Runs a fixed number of games for a given control type and returns the list of outcomes.
    [AI Declaration]: Generated using Claude with the prompt: "run the claw game and determine whether the player wins or loses"
    Args:
        control_type (str): The control mode, either "hand" or "keyboard".
        num_games (int): Number of games to run.
    Returns:
        list: List of outcomes where True=win, False=loss, None=quit.
    """
    results = []
    for i in range(num_games):
        print(f"\n--- {control_type.upper()} game {i + 1}/{num_games} ---")
        outcome = run_one_game_subprocess(control_type)
        results.append(outcome)
        if outcome is True:
            print("  -> Win")
        elif outcome is False:
            print("  -> Loss")
        else:
            print("  -> Quit (counted as loss for accuracy)")
        time.sleep(5.0)
    return results


def compute_metrics(results):
    """
    [Short Description]: Calculates win count, game count, accuracy percentage, and quit count from a list of outcomes.
    Args:
        results (list): List of outcomes where True=win, False=loss, None=quit.
    Returns:
        tuple: (wins, n, accuracy, quits) where accuracy is wins/n as a percentage and quits count as losses.
    """
    completed = [r for r in results if r is not None]
    quits = sum(1 for r in results if r is None)
    wins = sum(1 for r in results if r is True)
    n = len(results)
    # Accuracy = percent of games that were wins (out of all n games; quits count as loss)
    accuracy = (wins / n * 100.0) if n else 0.0
    return wins, n, accuracy, quits


def main():
    """
    [Short Description]: Entry point that runs the full evaluation, prints a summary, and logs results to txt and json files.
    Args:
        None
    Returns:
        None
    """
    print("Claw machine evaluation: keyboard vs hand tracking")
    print("You will play 5 games with KEYBOARD (arrow keys + space), then 5 with HAND (webcam).")
    user_name = input("Enter your name or id (for logging): ").strip() or "anonymous"

    print("\n" + "=" * 50)
    print("PART 1: KEYBOARD CONTROL (5 games)")
    print("Arrow keys or WASD: move | Space: grab | Q: quit game")
    print("=" * 50)
    keyboard_results = run_evaluation_round("keyboard", GAMES_PER_MODE)

    print("\n" + "=" * 50)
    print("PART 2: HAND TRACKING (5 games)")
    print("Open hand: move | Close fist: grab | Q: quit game")
    print("=" * 50)
    hand_results = run_evaluation_round("hand", GAMES_PER_MODE)

    # Metrics
    kb_wins, kb_n, kb_acc, kb_quits = compute_metrics(keyboard_results)
    hand_wins, hand_n, hand_acc, hand_quits = compute_metrics(hand_results)
    accuracy_diff = hand_acc - kb_acc  # positive = hand tracking did better

    summary = {
        "user": user_name,
        "date": datetime.now().isoformat(),
        "keyboard": {
            "wins": kb_wins,
            "games": kb_n,
            "accuracy_percent": round(kb_acc, 1),
            "quits": kb_quits,
        },
        "hand_tracking": {
            "wins": hand_wins,
            "games": hand_n,
            "accuracy_percent": round(hand_acc, 1),
            "quits": hand_quits,
        },
        "comparison": {
            "accuracy_diff_percent": round(accuracy_diff, 1),  # hand - keyboard
            "hand_better": hand_acc > kb_acc,
            "keyboard_better": kb_acc > hand_acc,
        },
    }

    # Console summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Keyboard:      {kb_wins}/{kb_n} wins -> {kb_acc:.1f}% accuracy" + (f" ({kb_quits} quit)" if kb_quits else ""))
    print(f"  Hand tracking: {hand_wins}/{hand_n} wins -> {hand_acc:.1f}% accuracy" + (f" ({hand_quits} quit)" if hand_quits else ""))
    print(f"  Difference:   Hand tracking is {accuracy_diff:+.1f}% vs keyboard")
    if accuracy_diff > 0:
        print("  -> Hand tracking performed better for you.")
    elif accuracy_diff < 0:
        print("  -> Keyboard performed better for you.")
    else:
        print("  -> Same accuracy for both.")

    # Append to text file
    with open(RESULTS_TXT, "a") as f:
        f.write("\n")
        f.write("-" * 60 + "\n")
        f.write(f"User: {user_name}  |  Date: {summary['date']}\n")
        f.write(f"  Keyboard:      {kb_wins}/{kb_n} wins -> {kb_acc:.1f}% accuracy")
        if kb_quits:
            f.write(f" ({kb_quits} quit)")
        f.write("\n")
        f.write(f"  Hand tracking: {hand_wins}/{hand_n} wins -> {hand_acc:.1f}% accuracy")
        if hand_quits:
            f.write(f" ({hand_quits} quit)")
        f.write("\n")
        f.write(f"  Difference:    Hand vs keyboard = {accuracy_diff:+.1f}%\n")
        f.write("-" * 60 + "\n")
    print(f"\nResults appended to {RESULTS_TXT}")

    # Append to JSON (one JSON array of all sessions, or append new line for easy parsing)
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
        if not isinstance(data, list):
            data = [data]
    else:
        data = []
    data.append(summary)
    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results appended to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
