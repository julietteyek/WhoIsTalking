import streamlit as st
import subprocess
import json
import re
import time
import random
import plotly.graph_objects as go

# sets the colors of the parties
party_colors = {
    "AfD": "#009ee0",
    "BÜNDNIS 90/DIE GRÜNEN": "#46962b",
    "CDU/CSU": "#32302e",
    "DIE LINKE": "#b61c3e",
    "FDP": "#ffed00",
    "SPD": "#e3000f"
}

# background for the Overlay-Bar
overlay_color = "#1c1c1c"

# background whole site
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc; /* Helle blaue Hintergrundfarbe */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# rename the parties for the legend
def rename_party_for_legend(party_name):
    if party_name == "BÜNDNIS 90/DIE GRÜNEN":
        return "DIE GRÜNEN"
    return party_name

# title
st.title("Who Is Talking? (beta)")
st.markdown("Gib einen Text ein, um zu sehen, welcher Partei er ähnelt!")

# text input
user_input = st.text_area("Dein Text", "")

# Button: start predicting..
if st.button("Vorhersage starten"):
    if user_input.strip():
        # animation for "Analysiere den Text..."
        message_placeholder = st.empty()
        for _ in range(3):
            message_placeholder.markdown("**Analysiere den Text**")
            time.sleep(0.4)
            message_placeholder.markdown("**Analysiere den Text.**")
            time.sleep(0.4)
            message_placeholder.markdown("**Analysiere den Text..**")
            time.sleep(0.4)
            message_placeholder.markdown("**Analysiere den Text...**")
            time.sleep(0.4)

        try:
            result = subprocess.run(
                ["python", "prediction.py"],  # direction to prediction.py
                input=user_input, text=True, capture_output=True
            )

            # is the output empty?
            output = result.stdout.strip()
            if not output:
                st.error("Keine Daten von prediction.py erhalten. Bitte überprüfe das Skript.")
                st.stop()

            # extract the JSON-part of the output
            json_match = re.search(r"{.*}", output)
            if json_match:
                json_output = json_match.group(0)  # extract
                try:
                    predictions = json.loads(json_output)
                except json.JSONDecodeError as e:
                    st.error(f"Fehler beim Parsen der JSON-Ausgabe: {str(e)}")
                    st.stop()
            else:
                st.error("Keine JSON-Ausgabe von prediction.py gefunden.")
                st.stop()

            # is the format correct?
            if not isinstance(predictions, dict):
                st.error("Unerwartetes Ergebnisformat. Bitte überprüfe prediction.py.")
                st.stop()

            # sort parties and their probabilities (descending)
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            sorted_parties = [item[0] for item in sorted_predictions]
            sorted_probabilities = [item[1] for item in sorted_predictions]

            # get the max probablity Party
            max_party = sorted_parties[0]
            max_probability = sorted_probabilities[0]

            # show random phrase with the result:
            result_phrases = [
                f"Dieser Text gehört höchstwahrscheinlich zur Partei: **{max_party}** mit **{max_probability * 100:.2f}%**.",
                f"Huch - kommt das aus den Reihen der {max_party}?",
                f"Das klingt ganz nach der Handschrift von {max_party} mit {max_probability * 100:.2f}% Wahrscheinlichkeit!",
                f"Spannend! Dieser Text könnte mit {max_probability * 100:.2f}% von {max_party} stammen.",
                f"Oh! Das passt perfekt zu {max_party} ({max_probability * 100:.2f}%).",
                f"Ein echter Treffer: {max_party} scheint hier mit {max_probability * 100:.2f}% am Werk gewesen zu sein!",
                f"Wow, das schreit förmlich nach {max_party}! Ganze {max_probability * 100:.2f}% Wahrscheinlichkeit!"
            ]
            st.markdown(
                f"<div style='background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; font-weight: bold;'>{random.choice(result_phrases)}</div>",
                unsafe_allow_html=True
            )

            # make the fixated bar-chart
            fig = go.Figure()

            # cumulated bar chart (no texts)
            for party, prob in zip(sorted_parties, sorted_probabilities):
                text_color = "black" if party == "FDP" else "white"  # text color for FDP
                text_font_size = 14 if party == max_party else 12  # bigger font for the max_party
                fig.add_trace(go.Bar(
                    x=[prob],
                    y=["Parteien"],
                    orientation='h',
                    marker=dict(color=party_colors.get(party, "#cccccc")),
                    text=f"{party}: {prob * 100:.2f}%" if party == max_party else f"{prob * 100:.2f}%" if prob >= 0.05 else None,
                    textposition="inside",
                    textfont=dict(color=text_color, size=text_font_size, family="Helvetica", weight="bold"),
                    name=f"{party}",  # only the party name in the legend
                    hoverinfo="text",  # tooltop text: party, percentage
                    hovertext=f"{party}: {prob * 100:.2f}%"
                ))

            # layout of the final bar-chart
            fig.update_layout(
                barmode='stack',
                xaxis=dict(title="Wahrscheinlichkeit (%)", tickformat=".0%"),
                yaxis=dict(showticklabels=False),
                showlegend=False,
                height=300
            )

            # animation with an Overlay-Bar
            progress_placeholder = st.empty()
            for step in range(100, -1, -1):  # Overlay von links nach rechts "ziehen"
                overlay_width = step / 100  # Breite des Overlays
                overlay_fig = go.Figure(fig)  # Kopiere die ursprüngliche Bar-Chart

                # add Overlay hinzufügen (starting on the left)
                overlay_fig.add_trace(go.Bar(
                    x=[overlay_width],  # width of the Overlay
                    y=["Parteien"],
                    orientation='h',
                    marker=dict(color=overlay_color),  # background color
                    showlegend=False,
                    hoverinfo="skip"
                ))

                # show progression
                progress_placeholder.plotly_chart(overlay_fig, use_container_width=True)
                time.sleep(0.03)  # simulation of an animation

            # show the Bar-Chart
            progress_placeholder.plotly_chart(fig, use_container_width=True)

            # show legend with Streamlit (sorted by probability)
            sorted_legend_data = sorted(
                party_colors.items(),
                key=lambda x: sorted_predictions[sorted_parties.index(x[0])][1],
                reverse=True
            )

            # generate the legend row-wise
            cols1 = st.columns(len(sorted_legend_data))  # first row: party + color
            for i, (party, color) in enumerate(sorted_legend_data):
                with cols1[i]:
                    st.markdown(
                        f"<div style='width: 15px; height: 15px; background-color: {color}; display: inline-block;'></div> {rename_party_for_legend(party)}",
                        unsafe_allow_html=True
                    )

            cols2 = st.columns(len(sorted_legend_data))  # second row: percentage
            for i, (party, _) in enumerate(sorted_legend_data):
                with cols2[i]:
                    st.markdown(f"**{predictions[party] * 100:.2f}%**")

        except Exception as e:
            st.error(f"Fehler bei der Ausführung von prediction.py: {str(e)}")
            st.stop()
    else:
        st.warning("Bitte gib einen Text ein, bevor du die Vorhersage startest.")
