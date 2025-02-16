import argparse
import clean
import prepare
import train
import prediction
import application

parser = argparse.ArgumentParser(description="BERT-based classification of political texts.")
parser.add_argument("task", choices=["train", "predict", "application"], help="Was soll ausgeführt werden?")
parser.add_argument("--text", type=str, help="Text für die Vorhersage (nur für 'predict' erforderlich)")

args = parser.parse_args()

"""
if args.task == "train":
    print("Starte Training des Modells...")
    train.main()
    print("Training abgeschlossen.")
elif args.task == "predict":
    if not args.text:
        print("Bitte einen Text mit --text angeben.")
    else:
        print("Starte Vorhersage...")
        result = prediction.main(args.text)
        print("Vorhersage abgeschlossen.")
        print("Ergebnis:", result)
elif args.task == "application":
    print("Starte Streamlit Anwendung...")
    application.main()
    print("Webanwendung läuft.")
"""
