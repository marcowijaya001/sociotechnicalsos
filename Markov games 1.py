import numpy as np
import mesa

STOCK_CAPACITY = 100
BINS = [0, 20, 50, 80, 101]      # discretized resource-stock states
ACTIONS = [1, 3, 6]              # harvest levels: low, medium, high


def to_state(stock):
    return np.digitize(stock, BINS) - 1


class HarvesterAgent(mesa.Agent):
    """An agent that repeatedly chooses a harvest level to maximize its
    own cumulative reward, given only the shared resource state."""

    def __init__(self, model, epsilon=0.3, alpha=0.1, gamma=0.9):
        super().__init__(model)
        self.q = np.zeros((len(BINS) - 1, len(ACTIONS)))
        self.epsilon, self.alpha, self.gamma = epsilon, alpha, gamma
        self.cumulative_reward = 0.0

    def act(self, state):
        if self.random.random() < self.epsilon:
            return self.random.randrange(len(ACTIONS))
        return int(np.argmax(self.q[state]))

    def learn(self, state, action, reward, next_state):
        best_next = self.q[next_state].max()
        self.q[state, action] += self.alpha * (
            reward + self.gamma * best_next - self.q[state, action]
        )
        self.cumulative_reward += reward


class CommonsModel(mesa.Model):
    """Markov game: state = shared resource stock. Each step, every agent's
    action (harvest level) jointly determines the reward split AND the next
    state, coupling the agents' individual MDPs into one game."""

    def __init__(self, n_agents=2, regen_rate=0.15, epsilon=0.3, rng=None):
        super().__init__(rng=rng)
        self.stock = float(STOCK_CAPACITY)
        self.regen_rate = regen_rate
        HarvesterAgent.create_agents(model=self, n=n_agents, epsilon=epsilon)
        self.datacollector = mesa.DataCollector(model_reporters={"Stock": lambda m: m.stock})

    def step(self):
        self.datacollector.collect(self)
        state = to_state(self.stock)

        actions = {a: a.act(state) for a in self.agents}
        requested = sum(ACTIONS[actions[a]] for a in self.agents)
        scale = min(1.0, self.stock / requested) if requested > 0 else 0.0

        rewards, harvested_total = {}, 0.0
        for a in self.agents:
            actual = ACTIONS[actions[a]] * scale
            rewards[a] = actual
            harvested_total += actual

        remaining = max(0.0, self.stock - harvested_total)
        self.stock = min(
            STOCK_CAPACITY,
            remaining + self.regen_rate * remaining * (1 - remaining / STOCK_CAPACITY),
        )
        next_state = to_state(self.stock)

        for a in self.agents:
            a.learn(state, actions[a], rewards[a], next_state)


if __name__ == "__main__":
    model = CommonsModel(n_agents=2, regen_rate=0.15, epsilon=0.3, rng=7)
    n_steps = 2000
    for t in range(n_steps):
        # decay exploration over time so behavior converges
        for a in model.agents:
            a.epsilon = max(0.02, 0.3 * (1 - t / n_steps))
        model.step()

    stock_df = model.datacollector.get_model_vars_dataframe()
    print("Mean stock, first 500 vs last 500 steps:")
    print("  first 500:", round(stock_df["Stock"].iloc[:500].mean(), 2))
    print("  last 500: ", round(stock_df["Stock"].iloc[-500:].mean(), 2))

    for i, a in enumerate(model.agents):
        print(f"\nAgent {i} cumulative reward: {round(a.cumulative_reward, 1)}")
        print(f"Agent {i} learned policy (best action per state bin, actions={ACTIONS}):")
        print("  ", [ACTIONS[x] for x in np.argmax(a.q, axis=1)])