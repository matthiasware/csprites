def get_max_positions(d_img, d_mask):
    return d_img - d_mask + 1


def get_evenly_spaced_positions(d_img, d_mask, spacing=1):
    max_positions = get_max_positions(d_img, d_mask)
    return list(range(0, max_positions, spacing))


def left_right_paddings(n):
    p_l = n // 2
    p_r = p_l + n % 2
    return p_l, p_r


def centered_corner_position_idcs(n_positions, max_positions):
    """
        Calculate evenly spaced positions,
        when max positions are available.
        e.g
        available positions:
               [1,2,3,4,5,6,7]
        want 4 positions:
               [2,3,4,5]
        want 3 positions:
               [1, 3, 5]
        want 2 positions:
               [2, 5]
        want 1 position:
               [4]
        returns: list of idcs, the first idx is 0
    """
    len_partitions, rest = divmod(max_positions, n_positions)

    # pad
    pad_left, pad_right = left_right_paddings(rest)

    # use the center element of the partion if partition is uneven
    # or the left element from center if partition is even
    partition_center = len_partitions // 2 + len_partitions % 2

    p_from = pad_left + partition_center
    p_to = max_positions - pad_right + 1
    p_step = len_partitions
    #
    positions = list(range(p_from, p_to, p_step))
    assert len(positions) == n_positions

    positions = [p - 1 for p in positions]
    return positions


def centered_position_idcs(n_positions, max_positions, d_mask):
    assert d_mask % 2 == 1

    positions = centered_corner_position_idcs(n_positions, max_positions)

    offset = d_mask // 2
    positions = [p + offset for p in positions]
    return positions


def get_position_idcs_from_center(h_mask, w_mask, px, py):
    """
        h_mask: mask height
        w_mask: mask width
        px: center pixel x axis
        py: center pixel y axis
    """
    x0 = px - w_mask // 2
    y0 = py - h_mask // 2
    x1 = px + w_mask // 2 + 1
    y1 = py + h_mask // 2 + 1
    return x0, y0, x1, y1
