from tqdm import tqdm


class TqdmBar:
    def __init__(self, total_time, env, description):
        self.bar = tqdm(
            total=total_time,
            desc=description,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} [Time: {elapsed}<{remaining}]"
            ),
        )
        self.env = env

    def update_progress(self):
        last_time = 0
        while True:
            self.bar.update(self.env.now - last_time)
            last_time = self.env.now
            yield self.env.timeout(100)

    def close(self):
        self.bar.close()
