import numpy as np
import math
import numbers


TRAIN = 'train'


def get_strip(n_samples, length=10, width=5, noise=0.5):
  x0 = np.random.uniform(-length/2, length/2, n_samples)
  x1 = np.random.uniform(-width/2, width/2, n_samples)
  return np.stack((x1, x0)).T


def puncture(strip, holes=1, hole_radius=1, length=0, width=0, ratio=False, save=False):
  x_width = strip[:, 0]
  x_length = strip[:, 1]
  if width == 0:
    width = x_width.max() - x_width.min()
  if length == 0:
    length = x_length.max() - x_length.min()
  if isinstance(holes, int):
    centers = np.empty((holes, 2))
    x_length_ticks = np.linspace(x_length.min(), x_length.max(), holes + 1)
    for i in range(holes):
      centers[i] = np.mean(strip[np.logical_and(x_length >= x_length_ticks[i],
                           x_length < x_length_ticks[i+1])], axis=0)
  else:
    centers = np.array(holes)
  print(centers.shape)

  length = length / centers.shape[0]

  if isinstance(hole_radius, numbers.Number):
    hole_radius = hole_radius * np.ones(centers.shape)
  elif length(hole_radius) == 2:
    hole_radius = np.dot(np.ones(centers.shape), np.array(hole_radius))
  else:
    hole_radius = np.array(hole_radius)

  if ratio:
    hole_radius = hole_radius * np.array((width, length))

  for i in range(centers.shape[0]):
    choices = np.sum((strip - centers[i]) ** 2 / hole_radius[i] ** 2, axis=1) > 1
    strip = strip[choices]

  return strip


def roll(strip):
  ratio = 1
  x_width = strip[:, 0]
  x_length = strip[:, 1]
  x_length -= x_length.min()
  return np.stack((ratio * (x_length + 1.5) * np.sin(x_length + math.pi/2),
                   ratio * (x_length + 1.5) * np.cos(x_length + math.pi/2),
                   x_width)).T, x_length


def get_swiss_roll(n_samples, length=10, width=5, noise=0.5, save=False):
  std_end = 4 * math.pi
  std_length = 0.5 * (std_end * math.sqrt(std_end ** 2 + 1)
                      + math.log(std_end + math.sqrt(std_end ** 2 + 1)))
  ratio = length / std_length
  # create dataset
  phi = np.random.uniform(0 + math.pi/2, std_end + math.pi/2, size=n_samples)
  Z = width * np.random.rand(n_samples)
  X = ratio * phi * np.sin(phi)
  Y = ratio * phi * np.cos(phi)
  err = np.random.normal(0, scale=noise, size=(n_samples, 3))

  swiss_roll = np.array([X, Y, Z]).transpose() + err

  # check that we have the right shape
  print(swiss_roll.shape)
  if save:
    np.save(TRAIN + '/swiss_roll_{}.npy'.format(n_samples), swiss_roll)
    np.save(TRAIN + '/swiss_roll_{}_positions.npy'.format(n_samples), phi)
  return swiss_roll, phi


def get_hole_centers_index(n_holes, n_samples):
  bounds = np.linspace(0, n_samples - 1, n_holes + 1)
  return ((bounds[:-1] + bounds[1:]) / 2).astype(int)


def poke_holes(swiss_roll, positions, holes=1, hole_radius=0, ratio=False, save=False):
  """
  :param swiss_roll: roll to poke holes at
  :param positions: relative position on the roll, also colors
  :param holes: if int: no of holes, if np.array: list of holes
  :param hole_radius: if 0: auto = 0.5 width of roll, if >0: radius of holes,
                      if pair: radii of ellipsoid holes, if np.array: list of radii
  :param ratio: if True: treat hole_radius as ratio to width of roll instead of real radii
  :param save: if True: save the roll with holes to a .npy file
  :return: the roll with holes and colors
  """
  order = np.argsort(positions)
  # print(order)
  positions_holes = positions[order]
  swiss_roll_holes = swiss_roll[order]
  z = swiss_roll_holes[:, 2]
  n_samples = positions_holes.shape[0]
  width = z.max() - z.min()
  strip = np.stack((z, positions_holes)).T
  # print(strip)

  if isinstance(holes, int):
    centers = strip[get_hole_centers_index(holes, n_samples)]
  else:
    centers = holes

  if hole_radius == 0:
    hole_radius = 0.5 * width * np.ones((centers.shape[0], 2))
  elif isinstance(hole_radius, float):
    hole_radius = hole_radius * np.ones((centers.shape[0], 2))
  elif isinstance(hole_radius, tuple):
    hole_radius = np.dot(np.ones((centers.shape[0], 2)), np.array(hole_radius))

  if ratio:
    hole_radius = hole_radius * width

  print('hole_radius.shape = {}'.format(hole_radius.shape))

  for p in centers:
    print(p)
    # print(strip)
    temp = np.dot((strip - p) ** 2, 1 / hole_radius.T ** 2)
    # print(temp)
    choices = temp > 1
    # print(np.concatenate((strip, temp, choices), axis=1))
    choices = choices.reshape(choices.shape[0])
    swiss_roll_holes = swiss_roll_holes[choices]
    positions_holes = positions_holes[choices]
  if save:
    np.save(TRAIN + '/swiss_roll_{}_holes.npy'.format(n_samples), swiss_roll_holes)
    np.save(TRAIN + '/swiss_roll_{}_holes_positions.npy'.format(n_samples), positions_holes)
  return swiss_roll_holes, positions_holes


# def get_swiss_roll_holes(n_samples, length=10, width=5, hole_radius=5, save=False):
#   swiss_roll, positions = get_swiss_roll(n_samples=n_samples, length=length, width=width, save=False)
#   centers = np.array([[-10., 10., 0.], [6., 10., 0.], [13., 10., 0.]])
#   swiss_roll_holes = swiss_roll
#   positions_holes = positions
#   for p in centers:
#     choices = np.sum((swiss_roll_holes - p) ** 2, axis=1) > hole_radius ** 2
#     swiss_roll_holes = swiss_roll_holes[choices]
#     positions_holes = positions_holes[choices]
#   if save:
#     np.save(TRAIN + '/swiss_roll_{}_holes.npy'.format(n_samples), swiss_roll_holes)
#     np.save(TRAIN + '/swiss_roll_{}_holes_positions.npy'.format(n_samples), positions_holes)
#   return



