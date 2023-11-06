from aalpy.SULs.AutomataSUL import VpaSUL, SevpaSUL
import random
import aalpy.utils.BenchmarkSevpaModels as SEVPAs
import aalpy.utils.BenchmarkVpaModels as VPAs

amount_languages = 15

missing_languages = {6}

sevpa_suls = []
vpa_suls = []
alphabets = []

for l in range(1, amount_languages+1):
    if l in missing_languages:
        sevpa_suls.append(None)
        vpa_suls.append(None)
        alphabets.append(None)
        continue

    language_sevpa = f'sevpa_for_L{l}'
    language_vpa = f'vpa_for_L{l}'

    # Get SEVPAs
    if hasattr(SEVPAs, language_sevpa):
        sevpa = getattr(SEVPAs, language_sevpa)()
    else:
        print(f"Function {language_sevpa} not found")
        continue
    sevpa_input_alphabet = sevpa.get_input_alphabet()
    sevpa_sul = SevpaSUL(sevpa, include_top=False, check_balance=False)
    sevpa_suls.append(sevpa_sul)
    alphabets.append(sevpa_input_alphabet)

    # Get VPA
    if hasattr(VPAs, language_vpa):
        vpa = getattr(VPAs, language_vpa)()
    else:
        print(f"Function {language_vpa} not found")
        continue
    vpa_input_alphabet = vpa.input_alphabet.get_merged_alphabet
    vpa_sul = VpaSUL(vpa, include_top=False, check_balance=False)
    vpa_suls.append(vpa_sul)

for l in range(0, amount_languages):
    language_index = l+1
    print(f'Checking Language L{language_index}')
    if language_index in missing_languages:
        print(f'Skipping L{language_index}')
        continue
    tests_passed = True
    for i in range(0, 100000):
        word_length = random.randint(1, 100)
        word = []
        for j in range(0, word_length):
            word.append(random.choice(alphabets[l]))

        pda_out = sevpa_suls[l].query(tuple(word))
        vpa_out = vpa_suls[l].query(tuple(word))

        if pda_out == vpa_out:
            continue
        else:
            print(f'Language L{language_index} failed on following test:')
            print(f'Input: {word}')
            print(f'Pda out: {pda_out} \nVpa out: {vpa_out}')
            tests_passed = False
            break

    if tests_passed:
        print(f'Language L{language_index} passed')
    else:
        print(f'Language L{language_index} failed')






