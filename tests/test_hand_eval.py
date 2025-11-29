# tests/test_hand_eval.py
from hand_eval import card_str_to_tuple, eval_5card

def as_cards(*cs):
    return [card_str_to_tuple(c) for c in cs]

def test_royal_flush():
    cat, tb = eval_5card(as_cards("AS","KS","QS","JS","TS"))
    assert cat == 9

def test_straight_flush_non_royal():
    cat, tb = eval_5card(as_cards("9S","8S","7S","6S","5S"))
    assert cat == 8 and tb == (9,)

def test_four_of_a_kind():
    cat, tb = eval_5card(as_cards("AH","AD","AS","AC","2D"))
    assert cat == 7 and tb[0] == 14

def test_full_house():
    cat, tb = eval_5card(as_cards("AH","AD","AS","KD","KC"))
    assert cat == 6 and tb == (14, 13)

def test_flush():
    cat, tb = eval_5card(as_cards("2S","8S","4S","9S","KS"))
    assert cat == 5

def test_straight_high_king():
    cat, tb = eval_5card(as_cards("9C","TC","JD","QH","KS"))
    assert cat == 4 and tb == (13,)

def test_wheel_straight():
    cat, tb = eval_5card(as_cards("AH","2D","3S","4C","5D"))
    assert cat == 4 and tb == (5,)

def test_trips():
    cat, tb = eval_5card(as_cards("AH","AD","AS","2C","3D"))
    assert cat == 3 and tb[0] == 14

def test_two_pair():
    cat, tb = eval_5card(as_cards("AH","AD","KC","KD","2S"))
    assert cat == 2 and tb[0] == 14 and tb[1] == 13

def test_one_pair():
    cat, tb = eval_5card(as_cards("JH","2D","JS","5C","9H"))
    assert cat == 1 and tb[0] == 11

def test_high_card():
    cat, tb = eval_5card(as_cards("2H","3D","5C","9H","KD"))
    assert cat == 0
