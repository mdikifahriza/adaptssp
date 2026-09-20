#pragma once

struct PairsTuple
{
    PairsTuple(size_t pairs_first_, size_t pairs_second_beg_, size_t pairs_n_second_, size_t pairs_offset_)
        : pairs_first(pairs_first_), pairs_second_beg(pairs_second_beg_), pairs_n_second(pairs_n_second_), pairs_offset(pairs_offset_) {}
    size_t pairs_first;
    size_t pairs_second_beg;
    size_t pairs_n_second;
    size_t pairs_offset;
};
