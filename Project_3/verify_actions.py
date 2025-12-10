# verify_actions.py
import time
import numpy as np
from inhand_env import CanRotateEnv
from actions_helper import discrete_to_continuous_action

# --- Main Verification Loop ---
def main():
    print("Initializing Environment (Human Render Mode)...")
    env = CanRotateEnv(render_mode="human")
    obs, info = env.reset()
    
    print("\n--- INSTRUCTIONS ---")
    print("Press '0' -> Open Hand")
    print("Press '1' -> Close Hand (Grasp)")
    print("Press '2' -> Rotate Clockwise")
    print("Press '3' -> Rotate Counter-Clockwise")
    print("Press 'q' -> Quit")
    print("--------------------")

    while True:
        user_input = input("Enter Action (0-3): ")
        
        if user_input.lower() == 'q':
            break
            
        try:
            action_idx = int(user_input)
            if action_idx not in [0, 1, 2, 3]:
                print("Invalid number. Use 0-3.")
                continue
                
            # Convert and Execute
            cont_action = discrete_to_continuous_action(action_idx)
            
            # We run the step multiple times (e.g. 10) to make the movement 
            # obvious to the human eye. Single step is too subtle.
            for _ in range(10): 
                obs, reward, done, truncated, info = env.step(cont_action)
                # Small sleep to smooth out rendering
                time.sleep(0.02) 
            
            print(f"Executed Action {action_idx}")
            
            if done or truncated:
                print("Episode ended. Resetting...")
                env.reset()
                
        except ValueError:
            print("Please enter a number.")
            
    env.close()

if __name__ == "__main__":
    main()