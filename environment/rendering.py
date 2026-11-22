"""Animated patient-navigation simulation (synthetic patients only).

The scene reads state and rewards from PatientNavigationEnv. It never changes
training logic, fabricates outcomes, or makes medical decisions.
"""

import random

import pygame

WIDTH, HEIGHT = 1280, 760
NAVY = (20, 42, 65)
BLUE = (37, 91, 130)
TEAL = (0, 142, 135)
GOLD = (236, 176, 67)
BG = (237, 244, 248)
WHITE = (255, 255, 255)
MUTED = (91, 113, 130)
RED = (184, 77, 74)
PALE = (207, 223, 233)

# A visual journey through support services; not a literal representation
# of the patient's physical movement or clinical treatment.
NODES = {
    "HOME": (160, 360),
    "OUTREACH": (386, 225),
    "TRANSPORT": (390, 500),
    "TELEHEALTH": (655, 225),
    "CLINIC": (655, 500),
    "CONNECTED": (885, 360),
}
ACTION_TARGETS = {
    0: "OUTREACH",  # Reminder
    1: "OUTREACH",  # Phone call
    2: "TRANSPORT",
    3: "TELEHEALTH",
    4: "HOME",      # No additional action
}


def choose_demo_action(observation, rng):
    """A random testing policy, NOT a trained RL policy."""
    return rng.randrange(5)


def target_for_action(action):
    return ACTION_TARGETS[int(action)]


def lerp(start, end, t):
    t = max(0.0, min(1.0, t))
    # Ease in/out, so the movement reads clearly in the demo.
    t = t * t * (3 - 2 * t)
    return (start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t)


def run_visual_demo(env, seed=42, action_selector=None, policy_name="DQN"):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("SmartCare RL | Animated Patient Navigation")
    clock = pygame.time.Clock()
    large = pygame.font.SysFont("segoeui", 36, bold=True)
    medium = pygame.font.SysFont("segoeui", 23, bold=True)
    body = pygame.font.SysFont("segoeui", 19)
    small = pygame.font.SysFont("segoeui", 16)
    tiny = pygame.font.SysFont("segoeui", 14)
    rng = random.Random(seed)
    observation = None
    history = []
    reward_total = 0.0
    finished = False
    status = ""
    current = NODES["HOME"]
    start = current
    destination = current
    moving = False
    move_start = 0
    pending = None
    last_action_time = 0

    def text(value, x, y, font=body, color=NAVY):
        screen.blit(font.render(str(value), True, color), (int(x), int(y)))

    def rounded(rect, color, radius=14):
        pygame.draw.rect(screen, color, pygame.Rect(rect), border_radius=radius)

    def reset_episode():
        nonlocal observation, history, reward_total, finished, status
        nonlocal current, start, destination, moving, move_start, pending
        observation, _ = env.reset(seed=seed)
        history = []
        reward_total = 0.0
        finished = False
        status = "Ready: patient begins at home"
        current = NODES["HOME"]
        start = current
        destination = current
        moving = False
        move_start = 0
        pending = None

    def begin_step(now):
        nonlocal pending, moving, move_start, start, destination, status
        if finished or moving:
            return
        action = (choose_demo_action(observation, rng) if action_selector is None
                  else int(action_selector(observation)))
        pending = action
        start = current
        destination = NODES[target_for_action(action)]
        moving = True
        move_start = now
        status = "Sending intervention: " + env.ACTION_NAMES[action]

    def complete_step():
        nonlocal observation, reward_total, finished, moving, current, status
        nonlocal pending, last_action_time
        observation, reward, terminated, truncated, info = env.step(pending)
        reward_total += reward
        history.append((env.steps, info["action"], float(reward)))
        current = destination
        moving = False
        pending = None
        last_action_time = pygame.time.get_ticks()
        if terminated or truncated:
            finished = True
            connected = bool(info.get("attended", False))
            status = "Appointment CONNECTED" if connected else "Appointment NOT connected"
            print(f"Episode finished | connected={connected} | total_reward={reward_total:+.2f}")
        else:
            status = "Action completed. Waiting for next intervention."

    reset_episode()
    last_action_time = pygame.time.get_ticks()
    automatic = True
    running = True
    move_ms = 1000

    try:
        while running:
            now = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        reset_episode()
                        rng.seed(seed)
                        last_action_time = now
                    elif event.key == pygame.K_SPACE:
                        automatic = not automatic
                        last_action_time = now
                    elif event.key == pygame.K_n and not automatic:
                        begin_step(now)

            if moving and now - move_start >= move_ms:
                complete_step()
            if automatic and not moving and not finished and now - last_action_time >= 950:
                begin_step(now)
                last_action_time = now

            # HEADER
            screen.fill(BG)
            rounded((0, 0, WIDTH, 104), NAVY, 0)
            text("SmartCare RL", 32, 19, large, WHITE)
            text("Patient navigation | Animated synthetic simulation", 34, 67, small, (199, 225, 236))
            rounded((1015, 28, 232, 44), (34, 68, 89), 11)
            text("RANDOM TEST POLICY" if action_selector is None else f"TRAINED {policy_name} POLICY",
                 1031, 40, small, (255, 213, 138))

            # MAIN JOURNEY PANEL
            rounded((24, 128, 936, 528), WHITE)
            text("Care access journey", 50, 146, medium)
            text("Follow each support action from the patient's starting point", 51, 181, small, MUTED)

            for a, b in [("HOME", "OUTREACH"), ("HOME", "TRANSPORT"),
                         ("OUTREACH", "TELEHEALTH"), ("TRANSPORT", "CLINIC"),
                         ("TELEHEALTH", "CONNECTED"), ("CLINIC", "CONNECTED")]:
                pygame.draw.line(screen, PALE, NODES[a], NODES[b], 8)
                pygame.draw.line(screen, (244, 248, 250), NODES[a], NODES[b], 3)

            labels = {
                "HOME": ("PATIENT", "Home / starting point"),
                "OUTREACH": ("OUTREACH", "Reminder or phone call"),
                "TRANSPORT": ("TRANSPORT", "Arrange travel support"),
                "TELEHEALTH": ("TELEHEALTH", "Remote access support"),
                "CLINIC": ("CLINIC", "In-person access route"),
                "CONNECTED": ("OUTCOME", "Connected or missed"),
            }
            for name, pos in NODES.items():
                if name == "HOME":
                    color = BLUE
                elif name == "CONNECTED":
                    color = TEAL if finished and env.steps == env.max_steps and history and status == "Appointment CONNECTED" else GOLD
                else:
                    color = (80, 142, 170)
                pygame.draw.circle(screen, (226, 237, 242), pos, 34)
                pygame.draw.circle(screen, color, pos, 25)
                pygame.draw.circle(screen, WHITE, pos, 11, 2)
                header, sub = labels[name]
                text(header, pos[0]-50, pos[1]+37, small, NAVY)
                text(sub, pos[0]-72, pos[1]+58, tiny, MUTED)

            # Token moves across the relevant support connection when action is applied.
            if moving:
                position = lerp(start, destination, (now - move_start) / move_ms)
            else:
                position = current
            pygame.draw.circle(screen, (255, 232, 164), (int(position[0]), int(position[1])), 20)
            pygame.draw.circle(screen, NAVY, (int(position[0]), int(position[1])), 11)
            pygame.draw.circle(screen, WHITE, (int(position[0]), int(position[1])), 4)

            rounded((45, 599, 887, 40), (233, 245, 247), 10)
            text(status, 59, 608, small, TEAL if status == "Appointment CONNECTED" else NAVY)

            # RIGHT SIDE: state, choices, feedback
            rounded((977, 128, 279, 528), WHITE)
            text("Patient profile", 997, 148, medium)
            phone, travel, telehealth, risk = env.profile
            details = [("Phone", "Yes" if phone else "No"),
                       ("Travel barrier", "Yes" if travel else "No"),
                       ("Telehealth", "Yes" if telehealth else "No"),
                       ("Missed-visit risk", f"{risk:.0%}")]
            for idx, (key, value) in enumerate(details):
                y = 187 + 37 * idx
                text(key, 999, y, small, MUTED)
                text(value, 1184, y, small, NAVY)
            pygame.draw.line(screen, PALE, (998, 342), (1235, 342), 2)
            text("Agent decisions", 998, 355, medium)
            for idx, (step, action_name, reward) in enumerate(history):
                y = 396 + idx * 55
                text(f"{step}. {action_name}", 997, y, tiny)
                text(f"Reward {reward:+.2f}", 999, y+20, small, TEAL if reward >= 0 else RED)
            if not history:
                text("Actions appear here...", 999, 409, small, MUTED)
            text(f"Total reward: {reward_total:+.2f}", 998, 577, body, NAVY)
            text(f"Steps: {env.steps}/{env.max_steps}", 998, 610, small, MUTED)

            rounded((24, 674, 1232, 63), NAVY, 12)
            text("SPACE: autoplay on/off     N: next action (manual)     R: restart     ESC: exit", 43, 687, small, WHITE)
            text(("Random demonstration only." if action_selector is None else f"Using learned {policy_name} decisions.")
                 + " Synthetic patient scenarios; no clinical decisions.",
                 43, 712, tiny, (204, 225, 235))
            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()
