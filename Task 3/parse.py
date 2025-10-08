#!/usr/bin/env python3

def calculate_average_score(file_path):
    try:
        with open(file_path, 'r') as file:
            scores = []
            for line in file:
                line = line.replace('\x00', '').strip()  # Rimuove i caratteri null e spazi bianchi
                if "Score" in line:
                    # Extract the score value
                    score = float(line.split(": ")[1])
                    scores.append(score)
            
            if scores:
                average_score = sum(scores) / len(scores)
                print(f"Average Score: {average_score:.6f} over {len(scores)} scores")
            else:
                print("No scores found in the file.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = "D:\\Luca\\università\\PAI\\Tasks\\Task 3\\task3_Docker\\scores.txt"  # Replace with your file name
calculate_average_score(file_path)
