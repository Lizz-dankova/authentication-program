import os
import time
import json
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

class KeyboardBiometrics:
    def __init__(self, training_iterations, user_id):
        self.training_iterations = training_iterations
        self.user_id = user_id
        self.biometric_data = []
        self.model = self.build_model()
        self.scaler = StandardScaler()
        self.typing_speed_stats = None
        self.hold_time_stats = None
        self.key_press_dynamics_stats = None

    def build_model(self):
        return SVR(kernel='linear')

    def collect_data(self):
        all_data = []
        for i in range(1, self.training_iterations + 1):
            start_time = time.time()
            phrase = input(f"Введіть фразу {i} для збору даних: ")
            end_time = time.time()
            hold_time = end_time - start_time
            typing_speed = len(phrase) / hold_time

            key_press_dynamics = self.measure_key_press_dynamics(phrase)

            data = {'user_id': self.user_id, 'phrase': phrase, 'phrase_number': i, 'typing_speed': typing_speed,
                    'hold_time': hold_time, 'key_press_dynamics': key_press_dynamics}
            all_data.append(data)
            self.biometric_data.append([self.user_id, i, typing_speed, hold_time, key_press_dynamics])

        filename = f'biometric_data_user_{self.user_id}.json'
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                json.dump(all_data, file, indent=4)
        else:
            with open(filename, 'r') as file:
                existing_data = json.load(file)
                existing_data.extend(all_data)
            with open(filename, 'w') as file:
                json.dump(existing_data, file, indent=4)

        biometric_data_array = np.array(self.biometric_data)
        self.scaler.fit(biometric_data_array[:, 2:])
        biometric_data_scaled = self.scaler.transform(biometric_data_array[:, 2:])

        X = biometric_data_scaled
        y = biometric_data_array[:, 3]
        scores = cross_val_score(self.model, X, y, cv=5)
        print("Cross-Validation Scores:", scores)
        print("Mean CV Score:", np.mean(scores))

        self.model.fit(X, y)
        self.calculate_intervals()

    def measure_key_press_dynamics(self, phrase):
        key_press_times = []
        for _ in phrase:
            key_press_times.append(time.time())
            time.sleep(0.1)

        avg_key_press_time = np.mean(np.diff(key_press_times)) if len(key_press_times) > 1 else 0
        return avg_key_press_time

    def identify(self):
        start_time = time.time()
        phrase = input("Введіть фразу для ідентифікації: ")
        end_time = time.time()
        hold_time = end_time - start_time
        typing_speed = len(phrase) / hold_time
        key_press_dynamics = self.measure_key_press_dynamics(phrase)

        lower_bound_speed, upper_bound_speed = self.typing_speed_stats
        lower_bound_hold, upper_bound_hold = self.hold_time_stats
        lower_bound_dynamics, upper_bound_dynamics = self.key_press_dynamics_stats

        print(f"Typing Speed - Lower Bound: {lower_bound_speed}, Upper Bound Speed: {upper_bound_speed}")
        print(f"Hold Time - Lower Bound: {lower_bound_hold}, Upper Bound Hold: {upper_bound_hold}")
        print(f"Key Press Dynamics - Lower Bound: {lower_bound_dynamics}, Upper Bound Dynamics: {upper_bound_dynamics}")
        print(f"Hold Time: {hold_time}, Typing Speed: {typing_speed}, Key Press Dynamics: {key_press_dynamics}")

        if lower_bound_hold <= hold_time <= upper_bound_hold:
            print("Час утримання не є аномальним.")
        else:
            print("Час утримання є аномальним.")

        if lower_bound_speed <= typing_speed <= upper_bound_speed:
            print("Швидкість набору не є аномальною.")
        else:
            print("Швидкість набору є аномальною.")

        if lower_bound_dynamics <= key_press_dynamics <= upper_bound_dynamics:
            print("Динаміка натискання клавіш не є аномальною.")
        else:
            print("Динаміка натискання клавіш є аномальною.")

        new_data = {
            'typing_speed': typing_speed,
            'hold_time': hold_time,
            'key_press_dynamics': key_press_dynamics
        }

        self.t_test_new_data(new_data)
        self.f_test_new_data(new_data)

        filename = f'biometric_data_user_{self.user_id}.json'
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                all_data = json.load(file)

                for data in all_data:
                    saved_typing_speed = data['typing_speed']
                    saved_hold_time = data['hold_time']
                    saved_key_press_dynamics = data['key_press_dynamics']

                    threshold_hold_time = 1
                    threshold_typing_speed = 1
                    threshold_key_press_dynamics = 0.1

                    hold_time_difference = abs(hold_time - saved_hold_time)
                    speed_difference = abs(typing_speed - saved_typing_speed)
                    dynamics_difference = abs(key_press_dynamics - saved_key_press_dynamics)

                    if (hold_time_difference < threshold_hold_time and
                            speed_difference < threshold_typing_speed and
                            dynamics_difference < threshold_key_press_dynamics):
                        print(f"Користувача {self.user_id} автентифіковано")
                        print("Швидкість набору:", typing_speed)
                        print("Speed difference", speed_difference)
                        print("Час утримання:", hold_time)
                        print("Hold time difference:", hold_time_difference)
                        print("Динаміка натискання клавіш:", key_press_dynamics)
                        print("Dynamics difference:", dynamics_difference)
                        return

                print(f"Користувач {self.user_id} - не користувач.")
        else:
            print(f"Файл з даними для користувача {self.user_id} не існує.")

    def calculate_intervals(self):
        filename = f'biometric_data_user_{self.user_id}.json'
        intervals_speed, intervals_hold, intervals_dynamics = [], [], []

        if os.path.exists(filename):
            with open(filename, 'r') as file:
                all_data = json.load(file)
                for data in all_data:
                    intervals_speed.append(data['typing_speed'])
                    intervals_hold.append(data['hold_time'])
                    intervals_dynamics.append(data['key_press_dynamics'])

            self.typing_speed_stats = self.calculate_bounds(intervals_speed)
            self.hold_time_stats = self.calculate_bounds(intervals_hold)
            self.key_press_dynamics_stats = self.calculate_bounds(intervals_dynamics)
        else:
            print(f"Файл з даними для користувача {self.user_id} не існує.")

    def calculate_bounds(self, intervals):
        mean = np.mean(intervals)
        std_deviation = np.std(intervals)
        lower_bound = mean - 5 * std_deviation
        upper_bound = mean + 5 * std_deviation

        print(f"Mean: {mean}, Std Deviation: {std_deviation}")
        print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

        return lower_bound, upper_bound

    def t_test_new_data(self, new_data):
        filename = f'biometric_data_user_{self.user_id}.json'
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                all_data = json.load(file)
                typing_speeds = [data['typing_speed'] for data in all_data]
                hold_times = [data['hold_time'] for data in all_data]
                key_press_dynamics = [data['key_press_dynamics'] for data in all_data]

            t_stat_speed, p_val_speed = ttest_ind(typing_speeds, [new_data['typing_speed']])
            t_stat_hold, p_val_hold = ttest_ind(hold_times, [new_data['hold_time']])
            t_stat_dynamics, p_val_dynamics = ttest_ind(key_press_dynamics, [new_data['key_press_dynamics']])

            print(f"t-statistic (Speed): {t_stat_speed}, p-value (Speed): {p_val_speed}")
            print(f"t-statistic (Hold Time): {t_stat_hold}, p-value (Hold Time): {p_val_hold}")
            print(f"t-statistic (Dynamics): {t_stat_dynamics}, p-value (Dynamics): {p_val_dynamics}")

            if p_val_speed < 0.05:
                print("Швидкість набору суттєво відрізняється від наявних даних.")
            else:
                print("Швидкість набору не суттєво відрізняється від наявних даних.")

            if p_val_hold < 0.05:
                print("Час утримання суттєво відрізняється від наявних даних.")
            else:
                print("Час утримання не суттєво відрізняється від наявних даних.")

            if p_val_dynamics < 0.05:
                print("Динаміка натискання клавіш суттєво відрізняється від наявних даних.")
            else:
                print("Динаміка натискання клавіш не суттєво відрізняється від наявних даних.")

    def f_test_new_data(self, new_data):
        filename = f'biometric_data_user_{self.user_id}.json'
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                all_data = json.load(file)
                typing_speeds = [data['typing_speed'] for data in all_data]
                hold_times = [data['hold_time'] for data in all_data]
                key_press_dynamics = [data['key_press_dynamics'] for data in all_data]

            f_stat_speed, p_val_speed = f_oneway(typing_speeds, [new_data['typing_speed']])
            f_stat_hold, p_val_hold = f_oneway(hold_times, [new_data['hold_time']])
            f_stat_dynamics, p_val_dynamics = f_oneway(key_press_dynamics, [new_data['key_press_dynamics']])

            print(f"F-statistic (Speed): {f_stat_speed}, p-value (Speed): {p_val_speed}")
            print(f"F-statistic (Hold Time): {f_stat_hold}, p-value (Hold Time): {p_val_hold}")
            print(f"F-statistic (Dynamics): {f_stat_dynamics}, p-value (Dynamics): {p_val_dynamics}")

            if p_val_speed < 0.05:
                print("Дисперсія швидкості набору суттєво відрізняється від наявних даних.")
            else:
                print("Дисперсія швидкості набору не суттєво відрізняється від наявних даних.")

            if p_val_hold < 0.05:
                print("Дисперсія часу утримання суттєво відрізняється від наявних даних.")
            else:
                print("Дисперсія часу утримання не суттєво відрізняється від наявних даних.")

            if p_val_dynamics < 0.05:
                print("Дисперсія динаміки натискання клавіш суттєво відрізняється від наявних даних.")
            else:
                print("Дисперсія динаміки натискання клавіш не суттєво відрізняється від наявних даних.")

    def save_additional_data(self):
        filename = f'biometric_data_user_{self.user_id}.json'
        all_data = []

        if os.path.exists(filename):
            with open(filename, 'r') as file:
                all_data = json.load(file)

            last_data_entry = all_data[-1] if all_data else None

            if last_data_entry:
                start_phrase_number = last_data_entry.get('phrase_number', 0) + 1
            else:
                start_phrase_number = 1
        else:
            start_phrase_number = 1

        new_data = []
        for i in range(start_phrase_number, start_phrase_number + self.training_iterations):
            start_time = time.time()
            phrase = input(f"Введіть фразу {i} для збору даних: ")
            end_time = time.time()
            hold_time = end_time - start_time
            typing_speed = len(phrase) / hold_time

            key_press_dynamics = self.measure_key_press_dynamics(phrase)

            data = {'user_id': self.user_id, 'phrase': phrase, 'phrase_number': i, 'typing_speed': typing_speed,
                    'hold_time': hold_time, 'key_press_dynamics': key_press_dynamics}
            new_data.append(data)

        all_data.extend(new_data)

        with open(filename, 'w') as file:
            json.dump(all_data, file, indent=4)

        self.calculate_intervals()

def main():
    training_iterations = 5

    while True:
        user_id = input("Введіть ідентифікатор користувача: ")
        filename = f'biometric_data_user_{user_id}.json'

        if not os.path.exists(filename):
            option = input("Такого користувача не існує. Бажаєте почати збирати дані?\n1. Почати збирати\n2. Назад до введення ідентифікатора\n3. Вихід\n")
            if option == "1":
                biometrics = KeyboardBiometrics(training_iterations, user_id)
                biometrics.collect_data()
            elif option == "2":
                continue
            elif option == "3":
                break
            else:
                print("Неправильний вибір опції.")
        else:
            biometrics = KeyboardBiometrics(training_iterations, user_id)
            biometrics.calculate_intervals()
            while True:
                option = input("Виберіть опцію:\n1. Доповнити дані існуючого користувача\n2. Автентифікувати існуючого\n3. Назад\n4. Вийти\n")
                if option == "1":
                    biometrics.save_additional_data()
                elif option == "2":
                    biometrics.identify()
                elif option == "3":
                    break
                elif option == "4":
                    return
                else:
                    print("Неправильний вибір опції.")

if __name__ == "__main__":
    main()
