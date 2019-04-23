import matplotlib.pyplot as plt
import numpy as np



#data = [4392, 8053, 12387, 16766, 20822, 24689, 28651, 32533, 36390, 40478, 44694, 48849, 53448, 57476, 61592, 65415, 69833, 73839, 78322, 82423, 87056, 91256]

# agents = 10 resources = 8
data2 = [[0.397, 0.592, 0.649, 0.709, 0.75, 0.782, 0.798, 0.816, 0.831, 0.842, 0.852, 0.857, 0.864, 0.866, 0.871, 0.875, 0.879, 0.882, 0.885, 0.886, 0.888, 0.89, 0.891, 0.892, 0.892, 0.891, 0.891, 0.891, 0.89, 0.89, 0.887, 0.884, 0.882, 0.879, 0.877, 0.874, 0.871, 0.869, 0.867, 0.866, 0.865, 0.864, 0.862, 0.861, 0.862, 0.862, 0.863, 0.862, 0.863, 0.863, 0.864, 0.865, 0.864, 0.865, 0.866, 0.867, 0.868, 0.868, 0.869, 0.87, 0.871, 0.872, 0.873, 0.873, 0.873, 0.873, 0.873, 0.874, 0.873, 0.872, 0.872, 0.871, 0.87, 0.869, 0.869, 0.868, 0.867, 0.867, 0.866, 0.865, 0.865, 0.865, 0.866, 0.866, 0.866, 0.867, 0.867, 0.868, 0.868, 0.868, 0.868, 0.868, 0.868, 0.868, 0.867, 0.867, 0.866, 0.867, 0.867, 0.866, 0.867, 0.867, 0.867, 0.866, 0.866, 0.866, 0.865, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.865, 0.865, 0.865, 0.865, 0.865, 0.866, 0.866, 0.866, 0.866, 0.865, 0.865, 0.865, 0.864, 0.864, 0.863, 0.863, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.864, 0.865, 0.865, 0.865, 0.865, 0.866, 0.866, 0.866, 0.866], [0.358, 0.583, 0.683, 0.748, 0.79, 0.819, 0.835, 0.85, 0.863, 0.872, 0.88, 0.886, 0.89, 0.891, 0.895, 0.897, 0.898, 0.9, 0.901, 0.902, 0.904, 0.906, 0.906, 0.907, 0.904, 0.902, 0.9, 0.897, 0.899, 0.9, 0.901, 0.902, 0.903, 0.903, 0.902, 0.903, 0.904, 0.904, 0.905, 0.905, 0.906, 0.906, 0.906, 0.906, 0.906, 0.906, 0.907, 0.905, 0.905, 0.905, 0.906, 0.906, 0.906, 0.906, 0.906, 0.905, 0.906, 0.906, 0.906, 0.907, 0.907, 0.907, 0.907, 0.907, 0.908, 0.908, 0.908, 0.907, 0.907, 0.907, 0.907, 0.907, 0.908, 0.908, 0.908, 0.909, 0.909, 0.909, 0.909, 0.909, 0.908, 0.907, 0.906, 0.905, 0.904, 0.903, 0.902, 0.901, 0.9, 0.899, 0.898, 0.898, 0.897, 0.895, 0.895, 0.894, 0.893, 0.893, 0.892, 0.891, 0.891, 0.89, 0.89, 0.89, 0.891, 0.891, 0.892, 0.892, 0.893, 0.893, 0.894, 0.894, 0.894, 0.894, 0.895, 0.895, 0.895, 0.896, 0.896, 0.896, 0.897, 0.897, 0.897, 0.898, 0.898, 0.898, 0.898, 0.899, 0.899, 0.899, 0.9, 0.9, 0.9, 0.901, 0.901, 0.901, 0.901, 0.901, 0.902, 0.902, 0.902, 0.902, 0.903, 0.903, 0.903, 0.903, 0.904, 0.904, 0.904, 0.904, 0.905, 0.905, 0.905, 0.905, 0.906, 0.906, 0.906], [0.548, 0.517, 0.508, 0.535, 0.549, 0.548, 0.552, 0.546, 0.545, 0.538, 0.535, 0.536, 0.536, 0.539, 0.541, 0.532, 0.531, 0.528, 0.527, 0.524, 0.524, 0.526, 0.527, 0.528, 0.527, 0.527, 0.527, 0.525, 0.525, 0.525, 0.523, 0.524, 0.526, 0.526, 0.525, 0.526, 0.526, 0.527, 0.526, 0.525, 0.525, 0.524, 0.523, 0.523, 0.524, 0.524, 0.524, 0.524, 0.524, 0.523, 0.523, 0.524, 0.525, 0.525, 0.525, 0.524, 0.523, 0.522, 0.523, 0.522, 0.523, 0.523, 0.522, 0.521, 0.52, 0.52, 0.519, 0.519, 0.519, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.521, 0.521, 0.521, 0.521, 0.522, 0.521, 0.522, 0.522, 0.522, 0.522, 0.521, 0.522, 0.523, 0.522, 0.522, 0.522, 0.523, 0.523, 0.522, 0.522, 0.523, 0.523, 0.523, 0.523, 0.522, 0.522, 0.522, 0.523, 0.522, 0.523, 0.523, 0.523, 0.523, 0.522, 0.522, 0.522, 0.522, 0.522, 0.522, 0.522, 0.522, 0.522, 0.522, 0.522, 0.522, 0.522, 0.522, 0.522, 0.522, 0.522, 0.523, 0.523, 0.523, 0.522, 0.523, 0.523, 0.522, 0.523, 0.522, 0.522, 0.522, 0.523, 0.522, 0.522, 0.522, 0.522, 0.522, 0.523, 0.523, 0.523, 0.523, 0.523, 0.523, 0.523, 0.522]]

# agents = 10 resources = 8 with migrating signal
data2 = [[0.252, 0.418, 0.507, 0.562, 0.6, 0.634, 0.659, 0.682, 0.698, 0.712, 0.725, 0.731, 0.74, 0.75, 0.76, 0.767, 0.772, 0.776, 0.78, 0.784, 0.79, 0.795, 0.801, 0.807, 0.811, 0.814, 0.818, 0.822, 0.824, 0.825, 0.825, 0.825, 0.827, 0.829, 0.829, 0.831, 0.83, 0.829, 0.828, 0.827, 0.827, 0.827, 0.828, 0.828, 0.828, 0.827, 0.826, 0.826, 0.827, 0.826, 0.827, 0.826, 0.826, 0.826, 0.827, 0.828, 0.829, 0.83, 0.831, 0.832, 0.833, 0.834, 0.834, 0.835, 0.836, 0.837, 0.839, 0.84, 0.841, 0.842, 0.842, 0.842, 0.842, 0.843, 0.843, 0.844, 0.845, 0.845, 0.845, 0.845, 0.845, 0.845, 0.845, 0.844, 0.844, 0.844, 0.844, 0.844, 0.843, 0.843, 0.843, 0.843, 0.843, 0.844, 0.844, 0.844, 0.844, 0.845, 0.845, 0.845, 0.844, 0.845, 0.844, 0.844, 0.843, 0.843, 0.843, 0.843, 0.843, 0.843, 0.843, 0.843, 0.843, 0.844, 0.844, 0.844, 0.844, 0.843, 0.843, 0.842, 0.843, 0.843, 0.844, 0.844, 0.844, 0.844, 0.845, 0.845, 0.846, 0.846, 0.847, 0.847, 0.847, 0.848, 0.848, 0.848, 0.849, 0.849, 0.848, 0.847, 0.847, 0.846, 0.846, 0.845, 0.845, 0.845, 0.845, 0.844, 0.844, 0.843, 0.842, 0.842, 0.842, 0.841, 0.841, 0.841, 0.84, 0.84, 0.839, 0.84, 0.839, 0.839, 0.84, 0.84, 0.84, 0.841, 0.841, 0.841, 0.841, 0.841, 0.841, 0.841, 0.841, 0.841, 0.841, 0.841, 0.841, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.841, 0.841, 0.841, 0.841, 0.841, 0.841, 0.84, 0.841, 0.841, 0.841, 0.84, 0.84, 0.84, 0.84, 0.839, 0.839, 0.839, 0.839, 0.839, 0.839, 0.839, 0.839, 0.839, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.841, 0.841, 0.841, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.842, 0.841, 0.841, 0.841, 0.841, 0.841], [0.238, 0.414, 0.52, 0.583, 0.624, 0.658, 0.685, 0.707, 0.723, 0.736, 0.746, 0.755, 0.767, 0.774, 0.783, 0.79, 0.795, 0.8, 0.803, 0.808, 0.813, 0.816, 0.821, 0.823, 0.824, 0.827, 0.829, 0.832, 0.836, 0.84, 0.842, 0.845, 0.847, 0.849, 0.852, 0.854, 0.856, 0.858, 0.86, 0.862, 0.863, 0.865, 0.867, 0.869, 0.87, 0.872, 0.873, 0.874, 0.876, 0.877, 0.878, 0.879, 0.88, 0.881, 0.882, 0.883, 0.884, 0.886, 0.887, 0.888, 0.889, 0.89, 0.891, 0.892, 0.893, 0.894, 0.895, 0.895, 0.896, 0.897, 0.897, 0.898, 0.899, 0.899, 0.9, 0.901, 0.901, 0.902, 0.902, 0.903, 0.903, 0.904, 0.904, 0.905, 0.905, 0.906, 0.906, 0.907, 0.908, 0.908, 0.908, 0.909, 0.909, 0.91, 0.91, 0.91, 0.911, 0.911, 0.911, 0.912, 0.912, 0.912, 0.913, 0.913, 0.913, 0.914, 0.914, 0.914, 0.915, 0.915, 0.915, 0.915, 0.916, 0.916, 0.916, 0.916, 0.916, 0.917, 0.916, 0.916, 0.916, 0.916, 0.916, 0.917, 0.917, 0.917, 0.917, 0.917, 0.917, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.919, 0.919, 0.919, 0.919, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.921, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.922, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.923, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.924, 0.925, 0.925, 0.925, 0.925, 0.925, 0.925, 0.925, 0.925, 0.925, 0.925, 0.925, 0.926, 0.926, 0.926, 0.926, 0.926, 0.926, 0.926, 0.926, 0.926, 0.926, 0.926, 0.926, 0.927, 0.927, 0.927, 0.927, 0.927, 0.927, 0.927, 0.927, 0.927, 0.927, 0.927, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928, 0.928], [0.432, 0.427, 0.452, 0.446, 0.454, 0.458, 0.463, 0.465, 0.458, 0.461, 0.468, 0.466, 0.471, 0.469, 0.468, 0.466, 0.464, 0.461, 0.461, 0.462, 0.463, 0.464, 0.463, 0.463, 0.464, 0.465, 0.465, 0.464, 0.464, 0.464, 0.466, 0.465, 0.465, 0.464, 0.465, 0.462, 0.462, 0.462, 0.461, 0.462, 0.461, 0.462, 0.461, 0.461, 0.461, 0.463, 0.464, 0.465, 0.464, 0.463, 0.464, 0.464, 0.464, 0.464, 0.464, 0.465, 0.465, 0.465, 0.465, 0.464, 0.465, 0.464, 0.464, 0.464, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.464, 0.465, 0.465, 0.465, 0.465, 0.466, 0.465, 0.465, 0.465, 0.465, 0.464, 0.464, 0.465, 0.464, 0.465, 0.465, 0.464, 0.464, 0.464, 0.464, 0.464, 0.465, 0.465, 0.465, 0.465, 0.466, 0.465, 0.465, 0.465, 0.464, 0.464, 0.464, 0.465, 0.464, 0.464, 0.464, 0.464, 0.464, 0.464, 0.464, 0.464, 0.463, 0.463, 0.463, 0.463, 0.463, 0.462, 0.462, 0.461, 0.461, 0.461, 0.461, 0.462, 0.461, 0.461, 0.461, 0.461, 0.461, 0.461, 0.461, 0.462, 0.462, 0.462, 0.462, 0.462, 0.461, 0.461, 0.462, 0.461, 0.461, 0.461, 0.461, 0.461, 0.461, 0.461, 0.461, 0.461, 0.461, 0.461, 0.461, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.464, 0.464, 0.464, 0.464, 0.464, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.463, 0.462, 0.463, 0.463, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462, 0.462]]

# agents = 10 resources = 15 with migrating signal
#data2 = [[0.221, 0.41, 0.502, 0.561, 0.598, 0.621, 0.639, 0.647, 0.663, 0.674, 0.685, 0.692, 0.702, 0.712, 0.718, 0.724, 0.728, 0.733, 0.737, 0.743, 0.747, 0.749, 0.75, 0.755, 0.758, 0.756, 0.756, 0.757, 0.758, 0.758, 0.757, 0.758, 0.757, 0.754, 0.751, 0.748, 0.745, 0.744, 0.744, 0.743, 0.743, 0.743, 0.743, 0.743, 0.743, 0.743, 0.744, 0.745, 0.746, 0.746, 0.747, 0.748, 0.749, 0.75, 0.751, 0.752, 0.752, 0.753, 0.753, 0.753, 0.753, 0.753, 0.752, 0.753, 0.754, 0.754, 0.755, 0.756, 0.757, 0.757, 0.757, 0.757, 0.757, 0.757, 0.756], [0.226, 0.408, 0.509, 0.549, 0.563, 0.586, 0.601, 0.617, 0.636, 0.646, 0.656, 0.661, 0.666, 0.671, 0.675, 0.682, 0.689, 0.695, 0.7, 0.706, 0.712, 0.716, 0.72, 0.724, 0.728, 0.731, 0.734, 0.738, 0.741, 0.743, 0.745, 0.747, 0.749, 0.75, 0.752, 0.753, 0.754, 0.756, 0.758, 0.759, 0.76, 0.761, 0.763, 0.764, 0.765, 0.766, 0.767, 0.768, 0.769, 0.769, 0.771, 0.772, 0.772, 0.773, 0.774, 0.774, 0.775, 0.776, 0.777, 0.777, 0.777, 0.778, 0.778, 0.779, 0.779, 0.78, 0.781, 0.782, 0.782, 0.783, 0.783, 0.784, 0.784, 0.785, 0.785], [0.403, 0.436, 0.442, 0.447, 0.452, 0.46, 0.469, 0.472, 0.469, 0.47, 0.467, 0.469, 0.47, 0.472, 0.474, 0.477, 0.477, 0.479, 0.479, 0.478, 0.477, 0.477, 0.478, 0.479, 0.48, 0.48, 0.478, 0.478, 0.478, 0.477, 0.475, 0.476, 0.476, 0.475, 0.474, 0.474, 0.475, 0.475, 0.476, 0.476, 0.475, 0.475, 0.473, 0.474, 0.474, 0.473, 0.473, 0.473, 0.474, 0.474, 0.475, 0.475, 0.475, 0.475, 0.474, 0.475, 0.475, 0.476, 0.476, 0.476, 0.476, 0.476, 0.476, 0.476, 0.475, 0.476, 0.476, 0.476, 0.475, 0.475, 0.475, 0.475, 0.474, 0.475, 0.475]]

# [0.955, 0.957, 0.957, 0.953, 0.947, 0.941, 0.933, 0.929, 0.922, 0.916, 0.91, 0.899, 0.892, 0.888, 0.883, 0.877, 0.865, 0.86, 0.858, 0.857, 0.858, 0.857, 0.856, 0.856, 0.858, 0.86, 0.863, 0.866, 0.868, 0.871, 0.874, 0.876, 0.879, 0.881, 0.883, 0.885, 0.887, 0.889, 0.89, 0.891, 0.892, 0.892, 0.891, 0.892, 0.893, 0.895, 0.896, 0.897, 0.898, 0.899, 0.899, 0.897, 0.896, 0.894, 0.894, 0.893, 0.891, 0.89, 0.889, 0.888, 0.886, 0.885, 0.883, 0.882, 0.879]# agents = 10 resources = 14 with migrating signal, channels
data2 = [[0.18, 0.395, 0.489, 0.544, 0.582, 0.604, 0.629, 0.645, 0.662, 0.676, 0.686, 0.692, 0.7, 0.706, 0.714, 0.721, 0.727, 0.731, 0.734, 0.737, 0.74, 0.742, 0.747, 0.752, 0.758, 0.763, 0.768, 0.771, 0.776, 0.78, 0.784, 0.787, 0.791, 0.795, 0.797, 0.8, 0.803, 0.805, 0.808, 0.81, 0.812, 0.813, 0.814, 0.816, 0.817, 0.819, 0.82, 0.822, 0.823, 0.823, 0.824, 0.825, 0.826, 0.827, 0.827, 0.827, 0.827, 0.826, 0.826, 0.825, 0.824, 0.823, 0.823, 0.822, 0.822, 0.822, 0.821, 0.821, 0.82, 0.819, 0.818, 0.818, 0.816, 0.815, 0.815, 0.814, 0.814, 0.814, 0.815, 0.816, 0.816, 0.815, 0.815, 0.814], [0.19, 0.39, 0.486, 0.532, 0.566, 0.591, 0.611, 0.628, 0.64, 0.65, 0.664, 0.678, 0.689, 0.7, 0.711, 0.72, 0.728, 0.733, 0.74, 0.744, 0.749, 0.753, 0.757, 0.76, 0.764, 0.767, 0.771, 0.773, 0.776, 0.778, 0.78, 0.782, 0.784, 0.786, 0.787, 0.789, 0.791, 0.792, 0.793, 0.794, 0.795, 0.796, 0.797, 0.798, 0.799, 0.799, 0.8, 0.801, 0.802, 0.803, 0.803, 0.804, 0.805, 0.805, 0.806, 0.806, 0.807, 0.807, 0.808, 0.809, 0.809, 0.81, 0.811, 0.812, 0.813, 0.814, 0.815, 0.816, 0.816, 0.817, 0.818, 0.819, 0.82, 0.82, 0.82, 0.821, 0.821, 0.822, 0.823, 0.824, 0.825, 0.825, 0.824, 0.824], [0.241, 0.225, 0.22, 0.225, 0.241, 0.252, 0.256, 0.252, 0.246, 0.242, 0.241, 0.24, 0.237, 0.234, 0.229, 0.226, 0.227, 0.224, 0.225, 0.226, 0.224, 0.221, 0.219, 0.219, 0.218, 0.218, 0.218, 0.217, 0.217, 0.216, 0.212, 0.208, 0.205, 0.202, 0.199, 0.199, 0.199, 0.2, 0.202, 0.202, 0.202, 0.201, 0.201, 0.201, 0.203, 0.207, 0.211, 0.214, 0.215, 0.217, 0.218, 0.218, 0.219, 0.219, 0.219, 0.219, 0.218, 0.217, 0.217, 0.216, 0.216, 0.215, 0.214, 0.213, 0.213, 0.213, 0.212, 0.211, 0.21, 0.21, 0.211, 0.21, 0.209, 0.208, 0.207, 0.207, 0.207, 0.207, 0.207, 0.207, 0.207, 0.206, 0.205, 0.205]]

time = np.arange(len(data2[0]))
#plt.plot(time, data)
plt.plot(time, data2[0],label="DQN")
plt.plot(time, data2[1],label="DDQN")
plt.plot(time, data2[2],label="RANDOM")
plt.legend()
plt.xlabel("time slot [1000]")
plt.show()
