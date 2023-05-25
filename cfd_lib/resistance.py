def get_loss_coefficient(open_area_ratio: float) -> float:
    """
    calculate the resistance coefficient based on open-area ratio
    """
    return 1 / (open_area_ratio**2.3) * (1 - open_area_ratio)


if __name__ == "__main__":
    print(get_loss_coefficient(0.001))
