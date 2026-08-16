from pathlib import Path
from zipfile import ZipFile

from pisteflow.sources.physiology import normalize_archive


def test_normalize_physiology_archive_preserves_upstream_signs(tmp_path: Path) -> None:
    archive_path = tmp_path / "physiology.zip"
    files = {
        "Participant Info.csv": (
            "ID,Gender,Age,Height,Weight,BMI,Wingspan,Fat mass %,Muscle mass %,"
            "Lean mass D arm %,Lean mass ND arm,Lean mass D leg,Lean mass ND leg\n"
            "1,F,20,170,60,20.8,172,15,45,3,2.8,9,8.9\n"
        ),
        "Core Temperature.csv": (
            "Participant,BL,Pre Poule,Post Poule,Difference,Pre DE,Post DE,Difference,Final\n"
            "1,37,37.1,37.2,0.1,37.2,37.4,0.2,37.3\n"
        ),
        "Bloods.csv": (
            "Lactate,,,,,,,,,,,Glucose,,\n"
            "Participant,BL1,BL2,BLAv,P1,P3,DE1,DE3,,,,Participant,BL,End\n"
            "1,1,1.2,1.1,2,2.2,3,3.2,,,,1,5,4.8\n"
        ),
        "RPE.csv": "Participant,Poule,DE,Final\n1,3,5,4\n",
        "Heart Rate.csv": (
            "Participant,P5_HRAV,P5_HRMAX,P5_HR%,DE_HRAV,DE_HRMAX,DE_HR%\n1,150,170,80,160,180,85\n"
        ),
        "Motion Analysis.csv": (
            "Participant,P total FT,DE total FT,P Active FT,DE Active FT,P Total RT,"
            "DE Total RT,P RT during F,DE RT during F,P RT bw bouts,DE RT bw bouts,"
            "Work:Rest (P),Work:rest (DE)\n"
            "1,10,20,5,8,5,12,4,10,1,2,1,1.5\n"
        ),
        "Gas Analysis.csv": (
            "Participant,P_VO2 abs,P_VO2 rel,P_VCO2,P_RER,DE_VO2 abs,DE_VO2 rel,"
            "DE_VCO2,DE_RER,EE (kcal.min)\n"
            "1,2,30,2,1,3,40,3,1.1,-12.5\n"
        ),
        "README.txt": "Licenses/restrictions placed on the data: CC-BY\n",
    }
    with ZipFile(archive_path, "w") as archive:
        for filename, payload in files.items():
            archive.writestr(filename, payload)

    frame = normalize_archive(archive_path)

    assert len(frame) == 1
    assert frame.loc[0, "weapon"] == "foil"
    assert frame.loc[0, "energy_expenditure_upstream_kcal_min"] == -12.5
    assert frame.loc[0, "core_temperature_de_difference_c"] == 0.2
