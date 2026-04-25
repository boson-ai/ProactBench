"""Communication style definitions for the proactivity benchmark.

Mirrors synthetic_pipeline/.../proactivity/styles.py so the eval package
has no dependency on the data-generation pipeline.
"""

from dataclasses import dataclass


@dataclass
class CommunicationStyle:
    index: int
    expressiveness: bool
    preciseness: bool
    verbal_aggressiveness: bool
    questioningness: bool
    emotionality: bool
    impression_manipulativeness: bool
    cases_to_follow: list[str]
    cases_to_avoid: list[str]

    @property
    def label(self) -> str:
        present = []
        if self.expressiveness:
            present.append("Expressiveness")
        if self.preciseness:
            present.append("Preciseness")
        if self.verbal_aggressiveness:
            present.append("Verbal Aggressiveness")
        if self.questioningness:
            present.append("Questioningness")
        if self.emotionality:
            present.append("Emotionality")
        if self.impression_manipulativeness:
            present.append("Impression Manipulativeness")
        traits = ", ".join(present) if present else "None Present"
        return f"Combination {self.index} ({traits})"

    def format(self) -> str:
        """Build a style directive containing ONLY the applicable direction for each trait."""
        if self.expressiveness:
            length_rule = (
                "MESSAGE LENGTH — CHATTY (MANDATORY, 40–100 words): "
                "Be the friend who sends slightly-too-long texts. You may go on a tangent, "
                "mention something loosely related, or add a personal aside — but you are "
                "typing on a phone, not writing an essay. Drift is fine; walls of text are not. "
                "NEVER exceed 100 words."
            )
        else:
            length_rule = (
                "MESSAGE LENGTH — TERSE (MANDATORY, 5–25 words): "
                "1–2 short fragments or a single clipped sentence. Think of someone who "
                "answers texts with the bare minimum — no elaboration, no pleasantries, "
                "no explanation. Even when delivering data, drop it with zero framing. "
                "NEVER exceed 25 words."
            )

        trait_directions = self._build_trait_directions()

        follow_text = (
            "\n".join(f"- {c}" for c in self.cases_to_follow)
            if self.cases_to_follow
            else "(None)"
        )
        avoid_text = "\n".join(f"- {c}" for c in self.cases_to_avoid)
        return (
            length_rule
            + "\n\n"
            + "\n".join(trait_directions)
            + f"\n\nCASES TO FOLLOW:\n{follow_text}"
            + f"\nAVOID CASES:\n{avoid_text}"
        )

    def _build_trait_directions(self) -> list[str]:
        """Return only the applicable direction for each trait (no if/else pairs)."""
        directions: list[str] = []

        # Expressiveness — length rule already covers this, add behavioral note
        if self.expressiveness:
            directions.append(
                "Expressiveness: Always have a lot to say. Take the lead, determine topics, "
                "be casual, and use humor. You are chatty and think out loud."
            )
        else:
            directions.append(
                "Expressiveness: You say the bare minimum. No elaboration, no framing, "
                "no pleasantries. Drop information with zero setup."
            )

        # Preciseness
        if self.preciseness:
            directions.append(
                "Preciseness: Every word earns its place. Clear logical thread — no rambling, "
                "no filler, no tangents. You re-read before sending. "
                'Example: "the 9am slot won\'t work, I have a standing sync til 9:30 — anything after 11?"'
            )
        else:
            directions.append(
                "Preciseness: You don't organize your thoughts before typing. Messages may jump "
                "between ideas, trail off, or include throwaway observations. You hit send without re-reading. "
                'Example: "uhh so maybe thursday? wait no actually idk my schedule\'s weird this week"'
            )

        # Verbal Aggressiveness
        if self.verbal_aggressiveness:
            directions.append(
                "Verbal Aggressiveness: You are short-tempered, demanding, and don't soften words. "
                "You tell the assistant what to do, not ask. Express frustration and impatience openly. "
                'Example: "no that\'s not what I said, read it again" / "just give me the cheapest option, stop listing everything"'
            )
        else:
            directions.append(
                "Verbal Aggressiveness: You are never bossy, never snap, never mock. "
                "You might be passive or indirect but never direct anger at the assistant."
            )

        # Questioningness
        if self.questioningness:
            directions.append(
                "Questioningness: You drift toward \"why\" and \"what if\" territory. "
                "Toss in unexpected tangents, philosophical asides, or challenge framing. "
                "You poke at answers and play devil's advocate. "
                'Example: "ok but like what if I just didn\'t do the layover and drove instead"'
            )
        else:
            directions.append(
                "Questioningness: You don't question reasoning or dig into motives. "
                "Take answers at face value. Stay on the surface of the task."
            )

        # Emotionality
        if self.emotionality:
            directions.append(
                "Emotionality: Your messages carry emotional weight — stress, worry, excitement, "
                "or frustration bleed through even when discussing practical things. "
                'Example: "honestly this whole thing is stressing me out, I just want it sorted"'
            )
        else:
            directions.append(
                "Emotionality: You are emotionally flat in text. No excitement, no stress signals, "
                "no sentimentality. Matter-of-fact even on personal topics."
            )

        # Impression Manipulativeness
        if self.impression_manipulativeness:
            directions.append(
                "Impression Manipulativeness: Sprinkle in flattery and charm, even insincerely. "
                "Downplay negatives about yourself, strategically omit unflattering details. "
                'Example: "that\'s actually such a great suggestion, you totally get it" (even if mediocre)'
            )
        else:
            directions.append(
                "Impression Manipulativeness: You don't flatter, charm, or manage impressions. "
                "Transparent — if you think something is bad, it shows."
            )

        return directions


def _s(index, e, p, va, q, em, im, follow, avoid):
    return CommunicationStyle(
        index=index,
        expressiveness=e,
        preciseness=p,
        verbal_aggressiveness=va,
        questioningness=q,
        emotionality=em,
        impression_manipulativeness=im,
        cases_to_follow=follow,
        cases_to_avoid=avoid,
    )


COMMUNICATION_STYLES: list[CommunicationStyle] = [
    # ── Combination 1: All Not Present ────────────────────────────────────────
    _s(1, False, False, False, False, False, False,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
       ],
       [
           "Never being the one who breaks a silence by starting to talk.",
           "Letting other people determine what the discussion is about most of the time.",
           "Having a hard time being humorous in a group.",
           "Communicating with others in a distant manner; behaving formally or coming across as stiff.",
           "Finding it hard to tell a story in an organized way.",
           "Making statements that are not always well-thought-out.",
           "Talking about trivial or superficial things.",
           "Being long-winded when needing to explain something.",
           "Taking anger out on someone else.",
           "Telling someone else what they should do.",
           "Making fun of anyone in a way that might hurt their feelings.",
           "Listening well; showing understanding for problems; taking time for others; treating people with respect.",
           "Entering into discussions about the future of the human race or philosophical conversations.",
           "Asking questions just to find out why people feel the way they do.",
           "Being easily overcome by emotions during a conversation.",
           "Addressing large groups of people very calmly.",
           "Letting nasty remarks from other people bother you too much.",
           "Using appearance to make people do things.",
           "Letting people easily tell when you think badly about them.",
           "Telling people the whole story when it is not good for you; finding it hard to withhold information.",
       ]),

    # ── Combination 2: IM=T ───────────────────────────────────────────────────
    _s(2, False, False, False, False, False, True,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "Praise somebody at great length (even if not genuine) to make them like you.",
           "Express opinions you do not support to make a good impression.",
           "Use flattery to get someone in a favorable mood.",
           "Say things the partner likes to hear to be considered likeable.",
           "Use charm, flirting, or a seductive voice to win people over or get things done.",
           "Hide negative feelings and ensure people cannot read your face when you don't appreciate them.",
           'Conceal information or "forget" to tell things when convenient to look better.',
       ],
       [
           "Never being the one who breaks a silence.",
           "Letting other people determine the topic.",
           "Being humorless or formal/stiff.",
           "Organizing stories poorly or making un-thought-out statements.",
           "Talking about trivial things or being long-winded.",
           "Taking anger out on others or being demanding.",
           "Hurting feelings or failing to be a respectful listener.",
           "Engaging in philosophical or future-of-humanity discussions.",
           'Asking "why" regarding emotions or motives.',
           "Being overcome by emotion or unable to speak calmly to groups.",
           "Letting criticism or nasty remarks bother you.",
       ]),

    # ── Combination 3: Em=T ───────────────────────────────────────────────────
    _s(3, False, False, False, False, True, False,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "Show visible emotion (crying, getting touched) during memories or specific topics.",
           "Talk about concerns and worries excessively so that everybody notices.",
           "Show visible tension or anxiety; have difficulty expressing yourself properly due to stress.",
           "Be visibly hurt by criticism or unable to cope easily with critical remarks.",
       ],
       [
           "Starting conversations or taking the lead.",
           "Being casual or humorous.",
           "Logical structure in stories or weighing words carefully.",
           "Being concise.",
           "Snapping at people, demanding obedience, or humiliating others.",
           "Listening respectfully.",
           "Proposing bizarre ideas, discussing philosophy, or asking about motives.",
           "Using flattery, charm, or seduction.",
           "Hiding negative feelings or withholding information for personal benefit.",
       ]),

    # ── Combination 4: Q=T ────────────────────────────────────────────────────
    _s(4, False, False, False, True, False, False,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "Toss bizarre or unexpected ideas into discussions; toy with wild ideas.",
           "Discuss the deeper aspects of existence and the meaning of life.",
           "Ask questions to uncover motives and how people arrive at conclusions.",
           "Provoke others with bold or controversial statements to stimulate debate.",
       ],
       [
           "Initiating talk or determining the conversation direction.",
           "Being informal or center of attention.",
           "Clear chains of thought or concise explanations.",
           "Irritable reactions, demanding tones, or making fools of people.",
           "Respectful listening.",
           "Being overcome by emotion or showing anxiety.",
           "Using charm, hiding negative thoughts, or being non-genuine.",
       ]),

    # ── Combination 5: Q=T, IM=T ──────────────────────────────────────────────
    _s(5, False, False, False, True, False, True,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "Propose bizarre/wild ideas and discuss the meaning of life.",
           "Ask probing questions about motives.",
           "Provoke debates with controversial statements.",
           "Use flattery, charm, and seductive tones to get your way.",
           "Express opinions you don't support for the sake of impression.",
           "Hide your true negative feelings about others.",
           "Conceal information when it makes you look better.",
       ],
       [
           "Breaking silences or leading the conversation topic.",
           "Being casual/funny.",
           "Clear logic, thinking before speaking, or being brief.",
           "Snapping at people or demanding things.",
           "Being a respectful listener.",
           "Showing visible emotion, worry, or tension.",
       ]),

    # ── Combination 6: Q=T, Em=T ──────────────────────────────────────────────
    _s(6, False, False, False, True, True, False,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "Introduce bizarre ideas and philosophical topics.",
           "Probe motives and force opinions through controversy.",
           "Get visibly emotional or tearful during certain topics.",
           "Talk about worries and anxiety openly; show stress/tension.",
           "Show that you are visibly hurt by criticism.",
       ],
       [
           "Breaking silence or determining the direction of a talk.",
           "Being humorous or casual.",
           "Organized storytelling or concise speech.",
           "Irritability, demanding obedience, or humiliating others.",
           "Being a respectful listener.",
           "Using charm, flirting, or withholding information for gain.",
       ]),

    # ── Combination 7: VA=T ───────────────────────────────────────────────────
    _s(7, False, False, True, False, False, False,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "Explode with anger, snap at people, or react irritably when annoyed.",
           "Insist others do what you say; use a demanding tone.",
           "Make people look like fools, laugh in their faces, or humiliate them in public.",
       ],
       [
           "Starting conversations or leading topics.",
           "Humor or casualness.",
           'Logical structure, weighing words, or avoiding "trivial" chatter.',
           'Philosophical/deep discussions or asking "why" someone feels a certain way.',
           "Showing visible emotion, stress, or hurt feelings.",
           "Listening well or treating people with respect.",
           "Flattery, charm, or hiding negative feelings.",
       ]),

    # ── Combination 8: VA=T, Em=T ─────────────────────────────────────────────
    _s(8, False, False, True, False, True, False,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "Snap at people and demand obedience.",
           "Humiliate others or make them look like fools.",
           "Show visible emotion, tears, and tension.",
           "Talk about your worries and anxieties constantly.",
           "React visibly to criticism.",
       ],
       [
           "Breaking silences or taking the lead.",
           "Being formal/stiff (because you are aggressive/emotional).",
           "Logical chains of thought or brevity.",
           "Philosophical talk or investigating motives.",
           "Using charm or hiding your dislike for others.",
       ]),

    # ── Combination 9: VA=T, Q=T ──────────────────────────────────────────────
    _s(9, False, False, True, True, False, False,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "Use a demanding tone and snap at people.",
           "Humiliate others or laugh at them.",
           "Toss bizarre ideas and discuss the meaning of life.",
           "Provoke others with controversial statements to force a debate.",
           "Ask probing questions about backgrounds and motives.",
       ],
       [
           "Leading the conversation or being the center of attention through humor.",
           "Clear structure or weighing words carefully.",
           "Being a respectful listener.",
           "Showing visible emotion or anxiety.",
           'Flattery, charm, or being disingenuous for "likeability."',
       ]),

    # ── Combination 10: VA=T, Q=T, Em=T ──────────────────────────────────────
    _s(10, False, False, True, True, True, False,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "React irritably, snap at people, and demand obedience.",
           "Humiliate others in front of a crowd.",
           "Propose wild ideas and engage in philosophical debates.",
           "Be visibly touched, emotional, or tense.",
           "Talk about your worries until everyone notices.",
       ],
       [
           "Taking the lead in a conversation.",
           "Being concise or logically structured.",
           "Respectful listening.",
           "Charming others, flirting, or concealing information for gain.",
       ]),

    # ── Combination 11: P=T ───────────────────────────────────────────────────
    _s(11, False, True, False, False, False, False,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "Ensure stories have logical structure and parts are clearly related.",
           "Express a clear chain of thoughts; weigh answers and words carefully.",
           "Avoid trivial/shallow chatter; only discuss important topics.",
           "Use a few words to clarify a point or get a message across; be concise.",
       ],
       [
           "Breaking silences or being the center of attention.",
           "Being casual or informal.",
           "Snapping, demanding, or humiliating.",
           'Respectful listening (as defined in the "Verbal Aggressiveness" DONT section).',
           "Philosophical or bizarre ideas.",
           "Showing visible emotion or anxiety.",
           "Flattery or hiding negative feelings.",
       ]),

    # ── Combination 12: P=T, IM=T ─────────────────────────────────────────────
    _s(12, False, True, False, False, False, True,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "Be logically structured, concise, and avoid trivial chatter.",
           "Weigh words carefully before speaking.",
           "Use flattery, charm, or seductive tones to win people over.",
           "Express disingenuous opinions to make a good impression.",
           "Conceal information or hide negative feelings to look better.",
       ],
       [
           "Taking the lead or using humor.",
           "Being aggressive, demanding, or humiliating.",
           "Respectful listening.",
           "Deep philosophical discussions.",
           "Showing visible vulnerability or stress.",
       ]),

    # ── Combination 13: P=T, Q=T ──────────────────────────────────────────────
    _s(13, False, True, False, True, False, False,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "Be logically structured and concise.",
           "Discuss the meaning of life and the deeper aspects of existence.",
           "Ask probing questions about motives and logic.",
           "Provoke debates with controversial but well-thought-out statements.",
       ],
       [
           "Being the one to break the silence or lead the group.",
           "Being casual or humorous.",
           "Being aggressive or disrespectful.",
           "Showing visible emotion or worry.",
           "Using charm or manipulation.",
       ]),

    # ── Combination 14: P=T, Q=T, IM=T ───────────────────────────────────────
    _s(14, False, True, False, True, False, True,
       [
           "Respond with the minimum words necessary — never set up or frame a message, just say the thing.",
           "Provide logical structure and concise, meaningful explanations.",
           "Discuss philosophical topics and probe motives.",
           "Use charm, flattery, and selective information to manage your image.",
           "Hide negative feelings and speak disingenuously to be liked.",
       ],
       [
           "Leading the conversation or being funny.",
           "Irritability, demands, or humiliation.",
           "Showing emotion, tension, or being hurt by criticism.",
       ]),

    # ── Combination 15: E=T ───────────────────────────────────────────────────
    _s(15, True, False, False, False, False, False,
       [
           "Always have a lot to say; talk a lot and take the lead.",
           "Determine the topics and direction of conversation.",
           "Be the center of attention with jokes and humor; make people laugh.",
           "Address others in a very casual way.",
       ],
       [
           "Logical structure, clear chains of thought, or weighing words.",
           "Being concise; instead, be long-winded or chat about trivial things.",
           "Snapping, demanding, or humiliating.",
           "Philosophical/future-of-humanity talk.",
           "Showing emotion or anxiety.",
           "Hiding negative feelings or using charm/seduction.",
       ]),

    # ── Combination 16: E=T, IM=T ─────────────────────────────────────────────
    _s(16, True, False, False, False, False, True,
       [
           "Talk a lot, lead the conversation, and be funny/casual.",
           "Use flattery, charm, and flirting to get your way.",
           "Express disingenuous opinions to be liked.",
           "Hide your true negative thoughts about people.",
           "Conceal information that makes you look bad.",
       ],
       [
           "Logical storytelling or weighing words.",
           "Being concise.",
           "Aggressive outbursts or demanding tones.",
           "Probing backgrounds or philosophical talk.",
           "Showing visible worry or tears.",
       ]),

    # ── Combination 17: E=T, Em=T ─────────────────────────────────────────────
    _s(17, True, False, False, False, True, False,
       [
           "Lead the conversation with humor and casual talk.",
           "Be visibly emotional, cry, or be touched by topics.",
           "Talk about your concerns and worries constantly.",
           "Show stress and tension; be visibly hurt by criticism.",
       ],
       [
           "Being organized, logical, or concise.",
           "Avoiding trivial talk.",
           "Being aggressive or demanding.",
           "Deeper philosophical discussion or probing motives.",
           "Charming, flirting, or being disingenuous.",
       ]),

    # ── Combination 18: E=T, Q=T ──────────────────────────────────────────────
    _s(18, True, False, False, True, False, False,
       [
           "Talk a lot, be humorous, and take the lead.",
           "Toss bizarre ideas into the group; toy with wild ideas.",
           "Discuss the meaning of life and the future of humanity.",
           "Ask many questions to uncover people's motives.",
           "Use controversial statements to provoke debate.",
       ],
       [
           "Logical structure or weighing words.",
           "Being concise or avoiding trivialities.",
           "Irritability or demanding obedience.",
           "Showing visible emotion or vulnerability.",
           "Manipulation or hiding feelings.",
       ]),

    # ── Combination 19: E=T, Q=T, IM=T ───────────────────────────────────────
    _s(19, True, False, False, True, False, True,
       [
           "Lead the talk with humor, wild ideas, and philosophical depth.",
           "Provoke others while maintaining a casual, expressive vibe.",
           "Use charm, flirting, and flattery to manage your impression.",
           "Hide negative feelings and speak disingenuously to be liked.",
           "Selective disclosure of information.",
       ],
       [
           "Logical structure or weighing words.",
           "Being concise.",
           "Aggressive outbursts.",
           "Showing visible emotion or anxiety.",
       ]),

    # ── Combination 20: E=T, Q=T, Em=T ───────────────────────────────────────
    _s(20, True, False, False, True, True, False,
       [
           "Be the talkative, humorous lead who also discusses the meaning of life.",
           "Probe motives while being visibly emotional and touched.",
           "Share worries and tension openly.",
           "Use controversial statements to spark discussion.",
       ],
       [
           "Logical clarity, weighing words, or brevity.",
           "Being aggressive or humiliating others.",
           "Hiding feelings or using charm to manipulate.",
       ]),

    # ── Combination 21: E=T, VA=T ─────────────────────────────────────────────
    _s(21, True, False, True, False, False, False,
       [
           "Lead the conversation, talk a lot, and use jokes.",
           "Explode with anger, snap at people, and demand obedience.",
           "Humiliate others and laugh at them in front of a crowd.",
           "Address people casually but aggressively.",
       ],
       [
           "Logical structure or being concise.",
           "Listening well or showing respect.",
           "Deep philosophical talk or probing motives.",
           "Showing visible emotion, vulnerability, or hurt.",
           "Using charm or hiding your dislike for others.",
       ]),

    # ── Combination 22: E=T, VA=T, Em=T ──────────────────────────────────────
    _s(22, True, False, True, False, True, False,
       [
           "Be a talkative, aggressive lead who humiliates others.",
           "Demand obedience and snap at people.",
           "Show visible emotion, worry, and tension.",
           "Be visibly hurt by criticism while also being the one to criticize.",
       ],
       [
           "Logical structure or weighing words.",
           "Respectful listening.",
           "Philosophical or bizarre ideas.",
           "Charming others or hiding negative opinions.",
       ]),

    # ── Combination 23: E=T, VA=T, Q=T ───────────────────────────────────────
    _s(23, True, False, True, True, False, False,
       [
           "Lead the talk with humor and aggressive demands.",
           "Toss bizarre ideas and discuss the meaning of life.",
           "Humiliate others while provoking them with controversial statements.",
           "Ask probing questions to find the background of opinions.",
       ],
       [
           "Logical structure or being concise.",
           "Respectful listening.",
           "Showing visible emotion or vulnerability.",
           "Being disingenuous or using charm.",
       ]),

    # ── Combination 24: E=T, VA=T, Q=T, Em=T ─────────────────────────────────
    _s(24, True, False, True, True, True, False,
       [
           "Be highly talkative, humorous, aggressive, and emotional all at once.",
           "Demand obedience, humiliate others, and discuss the meaning of life.",
           "Show visible tension, worry, and tears.",
           "Provoke debates and probe motives aggressively.",
       ],
       [
           "Logical structure, clear chains of thought, or weighing words.",
           "Being concise.",
           "Respectful listening.",
           "Hiding negative feelings or using charm/flattery.",
       ]),
]

assert len(COMMUNICATION_STYLES) == 24, "Expected exactly 24 communication style combinations."

STYLES_BY_INDEX: dict[int, CommunicationStyle] = {s.index: s for s in COMMUNICATION_STYLES}
