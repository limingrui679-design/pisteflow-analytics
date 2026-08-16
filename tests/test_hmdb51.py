from pisteflow.sources.hmdb51 import source_group, split_for_group


def test_source_group_keeps_sibling_clips_together() -> None:
    left = "Rapier_at_Clinton_2007_fencing_f_cm_np2_le_bad_0.avi"
    right = "Rapier_at_Clinton_2007_fencing_f_cm_np2_ri_bad_4.avi"

    assert source_group(left) == source_group(right)
    assert split_for_group(source_group(left)) == split_for_group(source_group(right))


def test_source_group_collapses_avi_name_variant() -> None:
    first = "Die_Another_Day_-_Fencing_Scene_Part_1_HD_avi_fencing_x.avi"
    second = "Die_Another_Day_-_Fencing_Scene_Part_1_HD_fencing_x.avi"

    assert source_group(first) == source_group(second)
