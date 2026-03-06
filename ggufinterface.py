import json
import re
from datetime import datetime
from llama_cpp import Llama
from vector_database import VectorDatabase
from pathlib import Path
import os
import argparse
import time
import textwrap

##### Color DEF
BLAU = "\033[94m"
GRÜN = "\033[92m"
ROT = "\033[91m"
GELB = "\033[93m"
RESET = "\033[0m"

MODE_SELFCHAT = "1"
MODE_META_ON = "2"
MODE_META_OFF = "3"
MODE_COMPARE = "4"

def hr(char="=", n=110):
    return char * n

def wrap_lines(text: str, width: int):
    lines = []
    for raw_line in text.splitlines() or [""]:
        if raw_line.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(raw_line, width=width, replace_whitespace=False, drop_whitespace=False))
    return lines


def print_two_columns(left_title: str, left_text: str, right_title: str, right_text: str, width=78, gap=4):
    left_lines = wrap_lines(left_text, width)
    right_lines = wrap_lines(right_text, width)
    max_len = max(len(left_lines), len(right_lines))
    left_lines += [""] * (max_len - len(left_lines))
    right_lines += [""] * (max_len - len(right_lines))

    title_line = f"{GELB}{left_title:<{width}}{' ' * gap}{right_title:<{width}}{RESET}"
    print(title_line)
    print(f"{GELB}{'-' * width}{RESET}{' ' * gap}{GELB}{'-' * width}{RESET}")

    for l, r in zip(left_lines, right_lines):
        print(f"{l:<{width}}{' ' * gap}{r:<{width}}")


class Chatbot:
    def __init__(
        self,
        gguf_filename,
        repo_id,
        directory,
        retriever=None,
        max_tokens=8192,
        max_new_tokens=2048,
        temperature=0.7,
        initial_message="Hallo! Wie kann ich helfen?",
        role_name="assistant",
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        seed=None,
        batch_size=64
    ):
        self.model = Llama.from_pretrained(
            repo_id,
            cache_dir="/home/klauth/workspace/ExpliCare/full_models/LLMs/LMStudio/",
            filename=gguf_filename,
            n_gpu_layers=-1,
            n_ctx=max_tokens,
            verbose=False,
            n_batch=batch_size,
            chat_format="chatml-function-calling",
        )

        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.role_name = role_name
        self.retriever = retriever
        self.history = [{"role": "system", "content": initial_message}]

        self.repo_id = repo_id
        self.gguf_filename = gguf_filename
        self.directory = f"{directory}/{self.repo_id}/"

        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.batch_size = batch_size

        self.tokens = self._recount_history_tokens()

    def count_tokens(self, text: str):
        return len(self.model.tokenize(text.encode("utf-8"), add_bos=True, special=False))

    def _recount_history_tokens(self):
        t = 0
        for msg in self.history:
            t += self.count_tokens(msg["content"]) + self.count_tokens(msg["role"]) + 10
        return t

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def clear_history(self):
        self.history = [{"role": "system", "content": self.history[0]["content"]}]
        self.tokens = self._recount_history_tokens()

    def print_history(self):
        print(f"{ROT}{hr()}{RESET}")
        print(f"{ROT}Aktuelle Chat-Historie:{RESET}")
        for msg in self.history:
            print(f"{ROT}{'-' * 55}{RESET}")
            if msg["role"].lower() == "user":
                print(f"{GRÜN}{msg['role'].capitalize()}: {RESET}{msg['content']}")
            else:
                print(f"{BLAU}{msg['role'].capitalize()}: {RESET}{msg['content']}")
        print(f"{ROT}{hr()}{RESET}")

    def print_tokens(self):
        print(f"{GELB}History Tokens: {self.tokens} / {self.max_tokens}{RESET}")

    def trim_history(self):
        while self.tokens >= self.max_tokens - self.max_new_tokens and len(self.history) > 2:
            rem = self.history.pop(1)
            self.tokens -= self.count_tokens(rem["content"]) + self.count_tokens(rem["role"]) + 10
            if len(self.history) > 2 and self.history[1]["role"] != "system":
                rem2 = self.history.pop(1)
                self.tokens -= self.count_tokens(rem2["content"]) + self.count_tokens(rem2["role"]) + 10

    def load_history(self, filename):
        file_path = os.path.join(self.directory + filename, f"{filename}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{ROT}Datei {filename}.json nicht gefunden in {self.directory}{RESET}")

        with open(file_path, "r", encoding="utf-8") as file:
            history = json.load(file) or []

        self.clear_history()

        # Rebuild history from the end until context budget is filled
        self.tokens = self._recount_history_tokens()
        for message in reversed(history):
            add_tok = self.count_tokens(message["content"]) + self.count_tokens(message["role"]) + 10
            if self.tokens + add_tok <= self.max_tokens - self.max_new_tokens:
                self.history.insert(1, message)
                self.tokens += add_tok
            else:
                break

        return self.history

    def append_to_json_file(self, filename):
        path = f"{self.directory + filename}/{filename}.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        existing_data.extend(self.history[-2:])

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

    def save_history(self, filename):
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename2 = f"{filename}_{timestamp}.json"
        filepath = Path(f"{self.directory}{filename}/{filename2}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)
        print(f"{GELB}Historie gespeichert in {filename2}{RESET}")

    def _build_base_instructions(self):
        return (
            "\n"
            "Der Nutzer hat eine Frage gestellt. Nutze die bereitgestellten Informationen und beantworte die Frage präzise und sachlich auf Deutsch.\n"
            "Schritte:\n"
            "1. Identifiziere das Hauptthema der Frage und prüfe die Kontextelemente auf Relevanz.\n"
            "2. Wähle die passendsten Kontextelemente aus. Ignoriere irrelevante Details und erwähne diese nicht.\n"
            "3. Formuliere eine prägnante, sachliche Antwort auf Deutsch, die die Frage bündig beantwortet und wenn möglich auf den bereitgestellten Informationen basiert.\n"
            "4. Falls die Informationen nicht ausreichend sind, antworte höflich mit deinem Wissen oder gib an, dass keine passenden Informationen vorliegen.\n"
            "5. Antworte immer so kurz wie möglich, ohne wichtige und erfragte Details auszulassen.\n"
            "\n"
        )

    def _build_augmented_contexts(self, prompt: str):
        patient_data = (
            "\n[Patientendaten]:\n"
            "[Name: N.N.\n"
            "Geburtsjahr: 1938\n"
            "Aufnahmedatum: 12.11.2025\n\n"
            "Anamnese & Hintergrund:\n"
            "Patient*in mit fortgeschrittener Demenz (MMST ca. 14/30). Orientierung stark eingeschränkt, zeitweise unruhig, benötigt vollständige Unterstützung bei der Mobilisation.\n\n"
            "Sturzereignisse:\n"
            "- Letzter dokumentierter Sturz am 07.11.2025 im Badezimmer, aus dem Stand.\n"
            "- Keine Frakturen, leichte Prellmarken am rechten Unterarm.\n"
            "- Seitdem engmaschige Sturzprophylaxe: rutschfeste Socken, Sensor-Matte, regelmäßige Toilettengänge unter Anleitung.\n\n"
            "Prophylaxen:\n"
            "- Letzte Sturz- und Dekubitusprophylaxe-Evaluierung: 11.11.2025\n"
            "- Mobilisation 3× täglich geplant, Lagerungsplan fortlaufend umgesetzt.\n\n"
            "Dekubitus:\n"
            "- Lokalisation: Sakralbereich\n"
            "- Stadium: I (nicht wegdrückbare Rötung)\n"
            "- Maßnahmen: Freilagerung mittels Wechseldruckmatratze, Hautpflege mit pH-neutraler Lotion, tägliche Kontrolle. Rötung aktuell rückläufig.\n\n"
            "Aktueller Zustand:\n"
            "Patient*in stabil, ruhig, nimmt Nahrung mit Unterstützung gut an. Hautzustand verbessert, keine neuen Sturzereignisse.]\n"
        )

        adl_trend = (
            "\n[Aktivitätsdaten]:\n"
            "[Erhebungszeitraum: 05.11.2025 – 12.11.2025\n"
            "Sensoren: Schlafzimmer-Bewegungssensor, Badezimmer-Aufenthaltszeit-Sensor\n\n"
            "05.11.2025:\n"
            "- Nächtliche Aktivität: gering, 2 kurze Bewegungsereignisse\n"
            "- Badezimmeraufenthalte: 3× tagsüber, jeweils < 5 Minuten\n"
            "- ADL Gesamteindruck: stabil\n\n"
            "07.11.2025:\n"
            "- Nächtliche Aktivität: moderat, 5 Bewegungsereignisse\n"
            "- Badezimmeraufenthalte: 4×, einmal 7 Minuten (Toilettengang mit Unterstützung)\n"
            "- ADL Gesamteindruck: leicht erhöhte Unruhe nach dokumentiertem Sturz\n\n"
            "09.11.2025:\n"
            "- Nächtliche Aktivität: gering\n"
            "- Badezimmeraufenthalte: 3×, unauffällig\n"
            "- ADL Gesamteindruck: ruhig, wenig Bewegungsdrang\n\n"
            "11.11.2025:\n"
            "- Nächtliche Aktivität: moderat, 4 Bewegungsereignisse\n"
            "- Badezimmeraufenthalte: 4×, zweimal ca. 6 Minuten\n"
            "- ADL Gesamteindruck: wach, aber kooperativ\n\n"
            "12.11.2025 (letzte Nacht):\n"
            "- Nächtliche Aktivität: deutlich erhöht, 11 Bewegungsereignisse im Schlafzimmer, teils längere Phasen von Umherwandern\n"
            "- Badezimmeraufenthalte: 2×, einer davon ca. 12 Minuten (verdacht auf Unruhe-bedingtes Ziellosigkeit)\n"
            "- ADL Gesamteindruck: ausgeprägte nächtliche Unruhe, ungewöhnlich erhöhte Bewegung\n\n"
            "Zusammenfassung:\n"
            "Die letzte Nacht zeigte ein klares Aktivitäts- und Unruhemuster. Empfehlung: Beobachtung intensivieren, ggf. Ursachen evaluieren (Schmerz, Angst, Schlafrhythmus, Umgebung).]\n"
        )

        expertise = ""
        distances_with_text = []
        if self.retriever:
            relevant_sentences = self.retriever.query(prompt, 5)
            db_context = ""
            for obj, dist, txt in relevant_sentences:
                db_context += txt + "\n"
                distances_with_text.append((txt, str(dist)))
            expertise = f"\n[Expertise]:\n[{db_context}]\n"

        contexts = {
            "patient_data": patient_data,
            "activity_data": adl_trend,
            "expertise": expertise,
        }

        token_stats = {
            "patient_data_tokens": self.count_tokens(patient_data),
            "activity_data_tokens": self.count_tokens(adl_trend),
            "expertise_tokens": self.count_tokens(expertise) if expertise else 0,
        }

        return contexts, token_stats

    # =========================
    # META LAYER PARSER (2 steps)
    # =========================

# Step 1
    def meta_step1_generate_meta_array(self, prompt: str, contexts: dict):
        meta_system = {
            "role": "system",
            "content": (
                "Analysiere die Nutzeranfrage und extrahiere die Metadaten. "
                "Antworte ausschließlich im JSON-Format.\n\n"
                "patient_data_required: ob [Patientendaten] benötigt werden.\n"
                "activity_data_required: ob [Aktivitätsdaten] benötigt werden.\n"
                "expertise_required: ob [Expertise] benötigt wird.\n"
            ),
        }

        meta_user = {
            "role": "user",
            "content": (
                (contexts.get("expertise") or "")
                + (contexts.get("activity_data") or "")
                + (contexts.get("patient_data") or "")
                + f"\nNutzeranfrage: {prompt}"
            ),
        }

        print(f"{BLAU}{hr('-')}{RESET}")
        print(f"{BLAU}META STEP 1/2: Meta-Array generieren (JSON){RESET}")
        print(f"{BLAU}{hr('-')}{RESET}")

        t0 = time.time()
        resp = self.model.create_chat_completion(
            messages=[meta_system, meta_user],
            max_tokens=256,
            stream=False,
            temperature=0,
            top_k=self.top_k,
            top_p=self.top_p,
            repeat_penalty=self.repetition_penalty,
            seed=self.seed,
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question_type": {
                            "type": "string",
                            "enum": ["Follow-up", "New question", "Small Talk"],
                            "description": (
                                "Klassifiziere, ob die Nutzeranfrage eine Folgefrage (bezieht sich auf vorherige Antwort/Details), "
                                "eine neue Frage oder Small Talk ist."
                            ),
                        },
                        "focus_topic": {
                            "type": "string",
                            "description": (
                                "Kurzer Fokus der Anfrage in 2-6 Wörtern (z.B. 'Sturzrisiko nachts', 'Dekubitus Stadium I', 'Ernährung Unterstützung')."
                            ),
                        },
                        "intent": {
                            "type": "string",
                            "enum": ["Information_request", "Create_action_plan", "Condition_assessment", "Risk_analysis", "Summary"],
                            "description": (
                                "Hauptziel der Anfrage: Info erfragen, Aktionsplan erstellen, Zustand bewerten, Risiko analysieren oder zusammenfassen."
                            ),
                        },
                        "urgency": {
                            "type": "string",
                            "enum": ["Routine", "Important", "Urgent"],
                            "description": (
                                "Dringlichkeit aus Sicht der Pflege/Medizin: "
                                "Routine=allgemein; Important=zeitnah prüfen; Urgent=sofortiges Handeln/Abklärung nötig."
                            ),
                        },

                        # ---- Decision flags ----
                        "patient_data_required": {
                            "type": "boolean",
                            "description": (
                                "Setze NUR True, wenn konkrete Infos aus [Patientendaten] nötig sind, um korrekt zu antworten.\n"
                                ),
                        },
                        "activity_data_required": {
                            "type": "boolean",
                            "description": (
                                "Setze NUR True, wenn Informationen aus [Aktivitätsdaten] (Sensor-/ADL-Verlauf, nächtliche Unruhe, Bewegungsmuster) "
                                "notwendig sind.\n"
                                "True-Beispiele:\n"
                                "- Fragen zu nächtlicher Unruhe, Umherwandern, Trend über Tage, Badezimmer-Aufenthaltszeiten, Veränderung im Verlauf.\n"
                                "- 'Wie war die letzte Nacht im Vergleich?' / 'Gibt es einen Trend zur Unruhe?' / 'Wie oft war die Person im Bad?'\n"
                            ),
                        },
                        "expertise_required": {
                            "type": "boolean",
                            "description": (
                                "Setze NUR True, wenn retrieved [Expertise] konkret benötigt wird.\n"
                                "True-Beispiele:\n"
                                "- Nutzer fragt nach evidenzbasierten Empfehlungen, Leitlinien, Dos & Don'ts, Warnzeichen, Standardmaßnahmen.\n"
                                "- Anfrage fordert spezifische fachliche Einordnung über den Fall hinaus.\n"
                            ),
                        },

                        "implicit_task": {
                            "type": "string",
                            "enum": ["Recommendation", "Information", "SmallTalk"],
                            "description": (
                                "Welche implizite Leistung soll erbracht werden: Empfehlung (Handlungsvorschläge), Information (Erklärung/Fakten), oder SmallTalk."
                            ),
                        },
                    },
                    "required": [
                        "question_type",
                        "focus_topic",
                        "intent",
                        "urgency",
                        "expertise_required",
                        "patient_data_required",
                        "activity_data_required",
                        "implicit_task",
                    ],
                },
            },

        )

        raw = resp["choices"][0]["message"]["content"]
        try:
            meta = json.loads(raw)
        except Exception:
            meta = {
                "question_type": "New question",
                "focus_topic": "Unklar",
                "intent": "Information_request",
                "urgency": "Routine",
                "expertise_required": False,
                "patient_data_required": False,
                "activity_data_required": False,
                "implicit_task": "Information",
            }

        dt = time.time() - t0
        print(f"{GRÜN}Meta-Array in {dt:.2f}s erzeugt:{RESET}")
        print(f"{GELB}{json.dumps(meta, ensure_ascii=False, indent=2)}{RESET}")
        return meta, dt

# Step 2
    def meta_step2_build_system_string(self, meta: dict, prompt: str):
        print(f"{BLAU}{hr('-')}{RESET}")
        print(f"{BLAU}META STEP 2/2: System-String aus Meta-Array erstellen{RESET}")
        print(f"{BLAU}{hr('-')}{RESET}")

        intention_string = f"""
Bei der aktuellen Anfrage handelt es sich um {"eine Folgefrage" if meta['question_type'] == 'Follow-up' else "eine neue Frage" if meta['question_type'] == 'New question' else "Small Talk"}.
Das Hauptthema der Anfrage ist: {meta['focus_topic']}.
Der Nutzer erfragt {"Informationen" if meta['intent'] == 'Information_request' else "einen Aktionsplan" if meta['intent'] == 'Create_action_plan' else "eine Zustandsbewertung" if meta['intent'] == 'Condition_assessment' else "eine Risikoanalyse" if meta['intent'] == 'Risk_analysis' else "eine Zusammenfassung"}.
Die Anfrage hat {"Routine-Priorität" if meta['urgency'] == "Routine" else "hohe Wichtigkeit" if meta['urgency'] == "Important" else "dringenden Charakter"}.
Für die Beantwortung wird {"spezifisches Fachwissen benötigt" if meta['expertise_required'] else "kein spezielles Fachwissen benötigt"}.
Es werden {"konkrete Patientendaten benötigt" if meta['patient_data_required'] else "keine Patientendaten benötigt"}.
Zusätzlich sind {"Aktivitätsdaten erforderlich" if meta['activity_data_required'] else "keine Aktivitätsdaten erforderlich"}.
Die implizite Aufgabe lautet: {"eine Empfehlung aussprechen" if meta['implicit_task'] == 'Recommendation' else "Informationen bereitstellen" if meta['implicit_task'] == 'Information' else "Small Talk führen"}.
Die Nutzeranfrage lautet: {prompt}
""".strip()

        print(f"{GRÜN}System-String erzeugt:{RESET}\n{GELB}{intention_string}{RESET}")
        return intention_string

    # =========================
    # PIPELINE: build prompt
    # =========================

    def _assemble_instruction_payload(self, prompt: str, base_instructions: str, contexts: dict, meta: dict | None):
        parts = [base_instructions]

        included = {"expertise": False, "patient_data": False, "activity_data": False}

        if meta is None:
            # Meta OFF: alles rein (Baseline)
            if contexts.get("expertise"):
                parts.append(contexts["expertise"])
                included["expertise"] = True
            parts.append(contexts["patient_data"])
            parts.append(contexts["activity_data"])
            included["patient_data"] = True
            included["activity_data"] = True
        else:
            # Meta ON: selektiv rein
            if meta.get("expertise_required") and contexts.get("expertise"):
                parts.append(contexts["expertise"])
                included["expertise"] = True
            if meta.get("patient_data_required"):
                parts.append(contexts["patient_data"])
                included["patient_data"] = True
            if meta.get("activity_data_required"):
                parts.append(contexts["activity_data"])
                included["activity_data"] = True

        parts.append(f"\nNutzeranfrage: {prompt}\n")

        payload = "".join(parts)
        return payload, included

    def _generate_response(self, messages, stream: bool):
        return self.model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            stream=stream,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repeat_penalty=self.repetition_penalty,
            seed=self.seed,
        )

    def _consume_output(self, output, stream: bool, start_time: float):
        if not stream:
            uncut_text = output["choices"][0]["message"]["content"]
            response_text = re.sub(r"^.*?<\|message\|>", "", uncut_text, flags=re.DOTALL).strip()
            return response_text, None

        told_time = False
        response_text = ""
        first_token_dt = None
        for chunk in output:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                if not told_time:
                    first_token_dt = time.time() - start_time
                    print(f"{GRÜN}Von Prompt zu 1. Token: {first_token_dt:.2f}s{RESET}")
                    told_time = True
                content = delta["content"]
                print(content, end="", flush=True)
                response_text += content
        return response_text, first_token_dt

    def run_single(self, prompt: str, filename: str, meta_enabled: bool, stream: bool = True, persist: bool = True):
        t_start = time.time()

        base_instructions = self._build_base_instructions()
        contexts, ctx_tok = self._build_augmented_contexts(prompt)

        print(f"{ROT}{hr()}{RESET}")
        print(f"{GELB}USER PROMPT:{RESET} {prompt}")
        print(f"{ROT}{hr()}{RESET}")

        print(f"{GELB}Kontext-Tokens (Roh):{RESET} "
              f"Patient={ctx_tok['patient_data_tokens']} | "
              f"Aktivität={ctx_tok['activity_data_tokens']} | "
              f"Expertise={ctx_tok['expertise_tokens']}")

        meta = None
        intention_string = None
        meta_dt = 0.0

    # META BONUS
        if meta_enabled:
            meta, meta_dt = self.meta_step1_generate_meta_array(prompt, contexts)
            intention_string = self.meta_step2_build_system_string(meta, prompt)
    # META BONUS 
        
        instruction_payload, included = self._assemble_instruction_payload(prompt, base_instructions, contexts, meta)

        payload_tokens = self.count_tokens(instruction_payload) + self.count_tokens("user") + 10
        print(f"{GELB}Payload-Tokens (an Modell):{RESET} {payload_tokens}")
        print(f"{GELB}Included Sections:{RESET} {included}")

        #print(f"{BLAU}{hr('-')}{RESET}")
        #print(f"{BLAU}KONTEXT / PAYLOAD (der an das Modell geht):{RESET}")
        #print(f"{BLAU}{hr('-')}{RESET}")
        #print(instruction_payload)

        working_messages = list(self.history) + [{"role": "user", "content": instruction_payload}]
    # META BONUS 
        if meta_enabled and intention_string:
            working_messages.append({"role": "system", "content": intention_string})
    # META BONUS 
        
        # token budget / trimming (nur anhand History grob)
        extra_tok = payload_tokens + (self.count_tokens(intention_string) + self.count_tokens("system") + 10 if meta_enabled and intention_string else 0)
        self.tokens += extra_tok
        self.trim_history()

        print(f"{ROT}{'-' * 55}{RESET}")
        print(f"{BLAU}{self.role_name.capitalize()}:{RESET}", end="")

    # NORMALE RESPONSE
        output = self._generate_response(working_messages, stream=stream)
        response_text, first_token_dt = self._consume_output(output, stream=stream, start_time=t_start)
    # NORMALE RESPONSE
    
        # revert token addition (wir speichern NICHT den payload in history)
        self.tokens -= extra_tok

        # commit clean chat history (nur prompt + antwort)
        self.add_message("user", prompt)
        self.tokens += self.count_tokens(prompt) + self.count_tokens("user") + 10

        self.add_message(self.role_name, response_text)
        self.tokens += self.count_tokens(response_text) + self.count_tokens(self.role_name) + 10

        if persist:
            self.append_to_json_file(filename)

        total_dt = time.time() - t_start
        print(f"\n{ROT}{'-' * 55}{RESET}")
        print(f"{GRÜN}Meta-Layer:{'AN' if meta_enabled else 'AUS'} | Meta-Time: {meta_dt:.2f}s | Gesamt: {total_dt:.2f}s{RESET}")

        return {
            "meta_enabled": meta_enabled,
            "meta": meta,
            "intention_string": intention_string or "",
            "included": included,
            "payload": instruction_payload,
            "payload_tokens": payload_tokens,
            "response": response_text,
            "meta_dt": meta_dt,
            "total_dt": total_dt,
            "first_token_dt": first_token_dt,
        }

    def run_compare(self, prompt: str, filename: str):
        print(f"{ROT}{hr()}{RESET}")
        print(f"{GELB}SIDE-BY-SIDE VERGLEICH (META AUS vs. META AN){RESET}")
        print(f"{ROT}{hr()}{RESET}")

        # Snapshot history: beide Runs gleiche Ausgangslage
        history_snapshot = list(self.history)
        tokens_snapshot = self.tokens

        # Run META OFF (no stream für sauberen Vergleich)
        self.history = list(history_snapshot)
        self.tokens = tokens_snapshot
        off = self.run_single(prompt, filename, meta_enabled=False, stream=False, persist=False)

        # Restore snapshot and run META ON
        self.history = list(history_snapshot)
        self.tokens = tokens_snapshot
        on = self.run_single(prompt, filename, meta_enabled=True, stream=False, persist=False)

        # Restore snapshot once more, dann committen wir nur META ON als "echte" Konversation
        self.history = list(history_snapshot)
        self.tokens = tokens_snapshot

        # Side-by-side: Kontext
        print(f"\n{ROT}{hr()}{RESET}")
        print(f"{GELB}KONTEXT / PAYLOAD VERGLEICH{RESET}")
        print(f"{ROT}{hr()}{RESET}")

        left = f"Payload Tokens: {off['payload_tokens']}\nIncluded: {off['included']}\n\n{off['payload']}"
        right = f"Payload Tokens: {on['payload_tokens']}\nIncluded: {on['included']}\n\n{on['payload']}"
        print_two_columns("META AUS", left, "META AN", right)

        # Side-by-side: Antwort
        print(f"\n{ROT}{hr()}{RESET}")
        print(f"{GELB}ANTWORT VERGLEICH{RESET}")
        print(f"{ROT}{hr()}{RESET}")

        left_r = f"Zeit: {off['total_dt']:.2f}s\n\n{off['response']}"
        right_r = f"Zeit: {on['total_dt']:.2f}s | Meta: {on['meta_dt']:.2f}s\n\n{on['response']}"
        print_two_columns("META AUS", left_r, "META AN", right_r)

        # Kurzes Ergebnis
        saved = off["payload_tokens"] - on["payload_tokens"]
        print(f"\n{GRÜN}Token-Ersparnis durch Meta-Layer: {saved} Tokens (positiv = weniger Kontext){RESET}")

        # Commit: wir behalten META AN als "echte" Antwort im Verlauf und speichern sie
        self.add_message("user", prompt)
        self.tokens += self.count_tokens(prompt) + self.count_tokens("user") + 10
        self.add_message(self.role_name, on["response"])
        self.tokens += self.count_tokens(on["response"]) + self.count_tokens(self.role_name) + 10
        self.append_to_json_file(filename)

    def _selfchat_next_question(self):
        self.add_message(
            "system",
            (
                "Du testes ein Chat-System. Dazu tust du so, als wärst du eine ungelernte Pflegekraft in der ambulanten Pflege, "
                "die eine Person pflegt und dazu einen Chat-Bot zur Hilfe nimmt. Du stellst Fragen bezüglich Dekubitus und Ernährung "
                "und Sturzereignissen. Immer eins nach dem anderen, dem aktuellen Kontext entsprechend. In dem Chat bist du 'user' "
                "und das System ist 'assistant'. Schreibe keine Einleitung, stelle ausschließlich eine Frage."
            ),
        )
        output = self.model.create_chat_completion(
            messages=self.history,
            max_tokens=self.max_new_tokens,
            stream=False,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repeat_penalty=self.repetition_penalty,
            seed=self.seed,
        )
        self.history.pop()

        uncut_text = output["choices"][0]["message"]["content"]
        response_text = re.sub(r"^.*?<\|message\|>", "", uncut_text, flags=re.DOTALL).strip()
        return response_text

    def chat(self):
        print("#######################################################################################")
        print("Chatbot ist bereit. Geben Sie Ihre Nachricht ein oder '/help' für Befehle.\n")

        print(
            f"{GELB}"
            "Demo-Modus wählen:\n"
            "  1) Self-Chat\n"
            "  2) Meta-Layer AN\n"
            "  3) Meta-Layer AUS\n"
            "  4) Side-by-Side Vergleich (Meta AUS vs. AN)\n"
            f"{RESET}",
            end="",
        )
        demo_mode = input("Auswahl (1/2/3/4): ").strip()

        filename = input("Bitte geben Sie den Namen des Chats ein: ").strip()
        filepath = Path(f"{self.directory}{filename}")
        filepath.mkdir(parents=True, exist_ok=True)

        files = [f for f in os.listdir(self.directory + filename) if f.endswith(".json")]
        if files:
            self.history = self.load_history(filename)
            print(f"{GRÜN}History von {filename} geladen.{RESET}")
            self.print_history()

        while True:
            print(f"\n{GRÜN}User: {RESET}", end="")

            if demo_mode == MODE_SELFCHAT:
                self.print_tokens()
                user_input = self._selfchat_next_question()
                print(user_input)
            else:
                user_input = input()

            cmd = user_input.strip().lower()
            if cmd == "/end":
                print("Chat beendet.")
                break
            if cmd == "/clear":
                self.clear_history()
                print("Chat-Historie wurde zurückgesetzt.")
                continue
            if cmd == "/load":
                print(f"{ROT}----------------Laden----------------{RESET}")
                load_filename = input("Bitte geben Sie den Namen des Chats ein: ").strip()
                try:
                    self.history = self.load_history(load_filename)
                    filename = load_filename
                    print(f"{GRÜN}History geladen.{RESET}")
                    self.print_history()
                except Exception as e:
                    print(f"{ROT}Fehler: {e}{RESET}")
                continue
            if cmd == "/history":
                self.print_history()
                continue
            if cmd == "/tokens":
                self.print_tokens()
                continue
            if cmd == "/help":
                print(f"{ROT}----------------Hilfe----------------{RESET}")
                print(
                    " /history  -> Chat-History anzeigen\n"
                    " /clear    -> History löschen\n"
                    " /end      -> Chat beenden\n"
                    " /tokens   -> Token-Status\n"
                    " /load     -> Anderen Chat laden\n"
                    " /help     -> Hilfe\n"
                )
                continue

            print(f"{ROT}{'-' * 55}{RESET}")

            if demo_mode == MODE_META_ON:
                self.run_single(user_input, filename, meta_enabled=True, stream=True, persist=True)
            elif demo_mode == MODE_META_OFF:
                self.run_single(user_input, filename, meta_enabled=False, stream=True, persist=True)
            elif demo_mode == MODE_COMPARE:
                self.run_compare(user_input, filename)
            else:
                # Fallback: Meta ON
                self.run_single(user_input, filename, meta_enabled=True, stream=True, persist=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Konfigurationsparameter für das Modell")

    parser.add_argument("--role_name", type=str, default="assistant")
    parser.add_argument("--model_name", type=str, default="MaziyarPanahi/Mistral-Nemo-Instruct-2407-GGUF")
    parser.add_argument("--filename", type=str, default="Mistral-Nemo-Instruct-2407.Q8_0.gguf")
    parser.add_argument("--model", type=str, default=None, choices=["nemo", "qwen", "llama", "mixtral", "phi"])
    parser.add_argument("--initial_message", type=str, default="Du bist ein hilfreicher Assistent, der nur auf deutsch Anfragen beantwortet.")
    parser.add_argument("--vectorize", default=False, action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_tokens", type=int, default=32000)
    parser.add_argument("--directory", type=str, default="./results/LM_Local/GGUF")
    parser.add_argument("--vectorizer_model_name", type=str, default="intfloat/multilingual-e5-large")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    repo_id = args.model_name
    gguf_filename = args.filename

    # Optional: bequeme Presets
    if args.model:
        if args.model == "qwen":
            repo_id = "Qwen/Qwen2.5-14B-Instruct-GGUF"
            gguf_filename = "qwen2.5-14b-instruct-q8_0-00001-of-00004.gguf"
        elif args.model == "nemo":
            repo_id = "MaziyarPanahi/Mistral-Nemo-Instruct-2407-GGUF"
            gguf_filename = "Mistral-Nemo-Instruct-2407.Q8_0.gguf"
        elif args.model == "mixtral":
            repo_id = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
            gguf_filename = "mixtral-8x7b-instruct-v0.1.Q4_0.gguf"
        elif args.model == "llama":
            repo_id = "MaziyarPanahi/calme-2.4-llama3-70b-GGUF"
            gguf_filename = "Llama-3-70B-Instruct-DPO-v0.4.Q2_K-00001-of-00006.gguf"

    embedder = VectorDatabase(
        model_name=args.vectorizer_model_name,
        output_dir="./results/vector_db/" + args.vectorizer_model_name,
        min_words=20,
        use_chatgpt=False,
    )
    if args.vectorize:
        embedder.build_index(evalu=False)
        embedder.save()
    embedder.load()

    bot = Chatbot(
        gguf_filename=gguf_filename,
        repo_id=repo_id,
        retriever=embedder,
        directory=args.directory,
        max_tokens=args.max_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        initial_message=args.initial_message,
        role_name=args.role_name,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    bot.chat()
