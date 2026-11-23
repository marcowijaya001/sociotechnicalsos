import mesa

class MoneyAgent(mesa.Agent):
    """An agent with one unit of wealth to start."""

    def __init__(self, model):
        super().__init__(model)
        self.wealth = 1

    def step(self):
        if self.wealth > 0:
            other_agent = self.random.choice(self.model.agents)
            other_agent.wealth += 1
            self.wealth -= 1


class MoneyModel(mesa.Model):
    """A model with N agents exchanging wealth at random."""

    def __init__(self, n=100, rng=None):
        super().__init__(rng=rng)
        self.num_agents = n
        MoneyAgent.create_agents(model=self, n=n)

        self.datacollector = mesa.DataCollector(
            model_reporters={"Gini": self.compute_gini},
            agent_reporters={"Wealth": "wealth"},
        )

    def compute_gini(self):
        wealths = sorted(a.wealth for a in self.agents)
        n = self.num_agents
        b = sum(w * (n - i) for i, w in enumerate(wealths)) / (n * sum(wealths))
        return 1 + (1 / n) - 2 * b

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")


if __name__ == "__main__":
    model = MoneyModel(n=100, rng=42)
    for _ in range(70):
        model.step()

    gini_df = model.datacollector.get_model_vars_dataframe()
    wealth_df = model.datacollector.get_agent_vars_dataframe()

    print("Gini coefficient, step 0, 10, 20, 30, 40, 49:")
    print(gini_df.iloc[[0, 10, 20, 30, 40, 50, 60, 69]])
    print("\nFinal wealth distribution (step 49):")
    print(wealth_df.xs(49, level="Step")["Wealth"].describe())
