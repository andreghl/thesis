from bargain import generate_observation
from bargain.routing.single import plot
import matplotlib.pyplot as plt


instance, routes, coalitions, gains, shapley, nucleolus = generate_observation(n_vehicles = 3,
                                                                               n_customers = 9,
                                                                               radius = 1.0,
                                                                               seed = 2026)

fig, axes = plt.subplot_mosaic(
    mosaic = [
        ["a", "a", "b", "b", "c", "c"],
        [".", "d", "d", "e", "e", "."]
    ],
    figsize = (18, 10),
    constrained_layout = True,
)

axes = [axes[k] for k in "abcde"]
coalitions = [[0, 1], [0, 2], [1, 2], [], [0, 1, 2]]
for ax, coalition in zip(axes, coalitions):
    plot(instance, routes, coalition, ax)

plt.show()

print(gains)
print(shapley)
print(nucleolus)