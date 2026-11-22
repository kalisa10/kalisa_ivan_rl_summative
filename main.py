"""Run the SmartCare patient-navigation simulator (terminal or visual mode)."""

import argparse

from environment import PatientNavigationEnv


def run_terminal(env):
    observation, _ = env.reset(seed=42)
    print("Patient navigation simulator - Stage 3")
    print("Initial observation:", observation)
    total_reward = 0.0
    finished = False
    while not finished:
        action = env.action_space.sample()  # Demonstration only: no learned agent yet.
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {env.steps}: {info['action']} | reward={reward:.2f}")
        finished = terminated or truncated
    print(f"Appointment connected: {info['attended']}")
    print(f"Total reward: {total_reward:.2f}")


def run_terminal_trained(env, model, greedy_action):
    observation, _ = env.reset(seed=42)
    finished = False
    score = 0.0
    while not finished:
        action = greedy_action(model, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        print(f"Step {env.steps}: {info['action']} | reward={reward:+.2f}")
        finished = terminated or truncated
    print(f"Appointment connected: {info['attended']} | total reward: {score:+.2f}")

def main():
    parser = argparse.ArgumentParser(description="SmartCare synthetic patient navigation")
    parser.add_argument("--visual", action="store_true", help="Display the Pygame simulation window")
    parser.add_argument("--visual3d", action="store_true", help="Launch the advanced Three.js 3D simulation")
    parser.add_argument("--port", type=int, default=8765, help="Local 3D visual server port")
    parser.add_argument("--algorithm", choices=["dqn", "reinforce", "ppo", "a2c"], default="dqn")
    parser.add_argument("--model", type=str, default=None,
                        help="Optional trained DQN checkpoint for the GUI")
    args = parser.parse_args()

    if args.visual3d:
        from visual3d.server import load_selector, make_handler
        from http.server import ThreadingHTTPServer
        import webbrowser
        selector = load_selector(args.algorithm if args.model else "random", args.model)
        label = args.algorithm if args.model else "random"
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(label, selector))
        url = f"http://127.0.0.1:{args.port}/"
        print(f"SmartCare Three.js 3D demo: {url} | Policy: {label.upper()}", flush=True)
        print("Press Ctrl+C to stop. Three.js library requires internet.", flush=True)
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return

    env = PatientNavigationEnv()
    try:
        if args.visual:
            from environment.rendering import run_visual_demo
            action_selector = None
            if args.model:
                if args.algorithm == "dqn":
                    from training.dqn_training import greedy_action, load_model
                    model = load_model(args.model)
                else:
                    from training.pg_training import greedy_action, load_policy
                    model = load_policy(args.algorithm, args.model)
                action_selector = lambda obs: greedy_action(model, obs)
            run_visual_demo(env, action_selector=action_selector, policy_name=args.algorithm.upper())
        else:
            if args.model:
                if args.algorithm == "dqn":
                    from training.dqn_training import greedy_action, load_model
                    model = load_model(args.model)
                else:
                    from training.pg_training import greedy_action, load_policy
                    model = load_policy(args.algorithm, args.model)
                run_terminal_trained(env, model, greedy_action)
            else:
                run_terminal(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
