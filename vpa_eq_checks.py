from aalpy.SULs.AutomataSUL import VpaSUL, PdaSUL
import random
import aalpy.utils.BenchmarkPdaModels as PDAs
import aalpy.utils.BenchmarkVpaModels as VPAs

amount_languages = 15

missing_languages = {6}

pda_suls = []
vpa_suls = []
alphabets = []

for l in range(1, amount_languages+1):
    if l in missing_languages:
        pda_suls.append(None)
        vpa_suls.append(None)
        alphabets.append(None)
        continue
    language_pda = f'pda_for_L{l}'
    language_vpa = f'vpa_for_L{l}'

    # Get PDAs
    if hasattr(PDAs, language_pda):
        pda = getattr(PDAs, language_pda)()
    else:
        print(f"Function {language_pda} not found")
        continue
    pda_input_alphabet = pda.get_input_alphabet()
    pda_sul = PdaSUL(pda, include_top=True, check_balance=True)
    pda_suls.append(pda_sul)
    alphabets.append(pda_input_alphabet)

    # Get VPA
    if hasattr(VPAs, language_vpa):
        vpa = getattr(VPAs, language_vpa)()
    else:
        print(f"Function {language_vpa} not found")
        continue
    vpa_input_alphabet = vpa.get_input_alphabet()
    merged_input_alphabet = vpa.get_input_alphabet_merged()
    vpa_sul = VpaSUL(vpa, include_top=True, check_balance=True)
    vpa_suls.append(vpa_sul)

for l in range(0, amount_languages):
    language_index = l+1
    print(f'Checking Language L{language_index}')
    if language_index in missing_languages:
        print(f'Skipping L{language_index}')
        continue
    tests_passed = True
    for i in range(0, 50000):
        word_length = random.randint(1, 100)
        word = []
        for j in range(0, word_length):
            word.append(random.choice(alphabets[l]))

        pda_out = pda_suls[l].query(tuple(word))
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





