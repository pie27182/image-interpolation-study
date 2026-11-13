import pytest

from interpolation_study.cli import build_parser, main


def test_top_level_help_exposes_commands_and_version(capsys):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    for option in ("--version", "resize", "benchmark", "plot", "plot-kernels", "methods"):
        assert option in output


def test_help_is_shown_when_no_command_is_given(capsys):
    main([])
    output = capsys.readouterr().out
    assert "usage:" in output
    assert "команды:" in output


def test_resize_help_contains_short_and_long_help_flags(capsys):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["resize", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "-h, --help" in output
    assert "--width" in output
    assert "--method" in output


def test_methods_command_lists_own_and_pillow_implementations(capsys):
    main(["methods"])
    output = capsys.readouterr().out
    assert "bspline3" in output
    assert "Pillow" in output
