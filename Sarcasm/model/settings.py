class Settings:
    model = "XGBoost"

    use_context = False
    use_author = False

    use_target_text = True
    use_target_audio = False
    use_target_video = True

    max_sent_length = 20
    max_context_length = 4
    num_classes = 2

    scale = True

    max_depth = [12, 6, 3]
    learning_rate = [0.1, 0.3, 0.5]
    gamma = [0, 0.005, 0.01]

    fold = None