# ============================================================
# EMOTION AUGMENTATION DATA
# ============================================================

augmentation_data = [

    # ========================================================
    # LOVE
    # ========================================================

    ("I love my family deeply.", "love"),
    ("I love my parents very much.", "love"),
    ("I care deeply about my family.", "love"),
    ("My family means so much to me.", "love"),
    ("I have so much love for my family.", "love"),
    ("I deeply care about my best friend.", "love"),
    ("My best friend means everything to me.", "love"),
    ("I have a deep affection for my friends.", "love"),
    ("I cherish the people in my life.", "love"),
    ("I feel a strong bond with my family.", "love"),
    ("I feel very close to my parents.", "love"),
    ("I love being around the people I care about.", "love"),
    ("My parents are very important to me.", "love"),
    ("I care so much about my brother.", "love"),
    ("I care so much about my sister.", "love"),
    ("I adore spending time with my family.", "love"),
    ("I have a special bond with my mother.", "love"),
    ("I have a special bond with my father.", "love"),
    ("I deeply love the people close to me.", "love"),
    ("My family is very dear to my heart.", "love"),
    ("I feel affection for my closest friends.", "love"),
    ("I love my friends more than words can say.", "love"),
    ("I feel a deep connection with my family.", "love"),
    ("I am grateful to have people I love around me.", "love"),
    ("I treasure my relationship with my family.", "love"),
    ("My best friend is very special to me.", "love"),
    ("I really care about my closest friends.", "love"),
    ("I love and appreciate my parents.", "love"),
    ("I feel deeply connected to the people I love.", "love"),
    ("The people I love are very important to me.", "love"),

    # ========================================================
    # JOY
    # ========================================================

    ("I am extremely happy today.", "joy"),
    ("I feel incredibly happy right now.", "joy"),
    ("Today has been wonderful.", "joy"),
    ("I am thrilled about my achievement.", "joy"),
    ("I am so excited about my success.", "joy"),
    ("I finally accomplished my goal.", "joy"),
    ("I feel fantastic today.", "joy"),
    ("I am delighted with the result.", "joy"),
    ("I am absolutely delighted.", "joy"),
    ("This is one of the happiest days of my life.", "joy"),
    ("I feel wonderful about what happened.", "joy"),
    ("I am really pleased with myself.", "joy"),
    ("I am celebrating my success.", "joy"),
    ("I feel proud and happy about my achievement.", "joy"),
    ("Everything went better than I expected.", "joy"),

    # ========================================================
    # SURPRISE
    # ========================================================

    ("I was completely shocked by the news.", "surprise"),
    ("That was totally unexpected.", "surprise"),
    ("I never imagined that would happen.", "surprise"),
    ("I never saw that coming.", "surprise"),
    ("The sudden announcement surprised me.", "surprise"),
    ("I was caught completely off guard.", "surprise"),
    ("I was shocked when I heard the news.", "surprise"),
    ("I cannot believe what just happened.", "surprise"),
    ("That completely caught me by surprise.", "surprise"),
    ("I was not expecting that at all.", "surprise"),
    ("The result was completely unexpected.", "surprise"),
    ("I was amazed by what happened.", "surprise"),
    ("I was stunned by the announcement.", "surprise"),
    ("That came as a huge surprise.", "surprise"),
    ("I had absolutely no idea this would happen.", "surprise"),
    ("I was genuinely surprised by the result.", "surprise"),
    ("The news caught me completely off guard.", "surprise"),
    ("I could not have predicted that.", "surprise"),
    ("That happened much sooner than I expected.", "surprise"),
    ("I was shocked to see what happened.", "surprise"),

    # ========================================================
    # SADNESS
    # ========================================================

    ("I feel deeply lonely today.", "sadness"),
    ("I feel like nobody understands me.", "sadness"),
    ("I feel empty inside.", "sadness"),
    ("I have been feeling miserable lately.", "sadness"),
    ("I feel emotionally drained.", "sadness"),
    ("I feel really down today.", "sadness"),
    ("I feel lost and alone.", "sadness"),
    ("I have been feeling very unhappy lately.", "sadness"),
    ("I feel like everything is going wrong.", "sadness"),
    ("I feel heartbroken about what happened.", "sadness"),
]


# ============================================================
# PRINT INFORMATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EMOTION AUGMENTATION DATA")
    print("=" * 60)

    print(f"Total additional examples: {len(augmentation_data)}")

    counts = {}

    for text, emotion in augmentation_data:
        counts[emotion] = counts.get(emotion, 0) + 1

    print("\nExamples by emotion:")

    for emotion, count in counts.items():
        print(f"{emotion:<10}: {count}")

    print("=" * 60)