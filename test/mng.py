# Scattermind distributes computation of machine learning models.
# Copyright (C) 2024 Josua Krause
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import collections
import os
import re
from collections.abc import Iterable
from xml.etree import ElementTree as ET


TEST_FILE_PATTERN = re.compile(r"^test_.*\.py$")
XML_EXT = ".xml"
DEFAULT_TEST_DURATION = 10.0


def listdir(folder: str) -> Iterable[str]:
    yield from sorted(os.listdir(folder))


def find_tests(folder: str) -> Iterable[str]:
    for item in listdir(folder):
        if not os.path.isdir(item) and TEST_FILE_PATTERN.match(item):
            yield os.path.join(folder, item)


def merge_results(base_folder: str, out_filename: str) -> None:
    testsuites = ET.Element("testsuites")
    combined = ET.SubElement(testsuites, "testsuite")
    parts_folder = os.path.join(base_folder, "parts")
    tests = 0
    skipped = 0
    failures = 0
    errors = 0
    time = 0.0
    for file_name in listdir(parts_folder):
        if not file_name.endswith(XML_EXT):
            continue
        tree = ET.parse(os.path.join(parts_folder, file_name))
        test_suite = tree.getroot()[0]

        combined.attrib["name"] = test_suite.attrib["name"]
        combined.attrib["timestamp"] = test_suite.attrib["timestamp"]
        combined.attrib["hostname"] = test_suite.attrib["hostname"]
        failures += int(test_suite.attrib["failures"])
        skipped += int(test_suite.attrib["skipped"])
        tests += int(test_suite.attrib["tests"])
        errors += int(test_suite.attrib["errors"])
        for testcase in test_suite:
            time += float(testcase.attrib["time"])
            ET.SubElement(combined, testcase.tag, testcase.attrib)

    combined.attrib["failures"] = f"{failures}"
    combined.attrib["skipped"] = f"{skipped}"
    combined.attrib["tests"] = f"{tests}"
    combined.attrib["errors"] = f"{errors}"
    combined.attrib["time"] = f"{time}"

    new_tree = ET.ElementTree(testsuites)
    with open(os.path.join(base_folder, out_filename), "wb") as fout:
        new_tree.write(fout, xml_declaration=True, encoding="utf-8")


def split_tests(filepath: str, total_nodes: int, cur_node: int) -> None:
    _, fname = os.path.split(filepath)
    base = "test"
    if not fname.endswith(XML_EXT):
        raise ValueError(f"{fname} is not an xml file")
    test_files = list(find_tests(base))
    time_keys: list[tuple[str, float]]
    try:
        tree = ET.parse(filepath)
        test_time_map: collections.defaultdict[str, float] = \
            collections.defaultdict(lambda: 0.0)
        for testcases in tree.getroot()[0]:
            cname = testcases.attrib["classname"].replace(".", "/")
            cname = f"{cname}.py"
            fname = os.path.normpath(cname)
            test_time_map[fname] += float(testcases.attrib["time"])

        for file in test_files:
            fname = os.path.normpath(file)
            if fname not in test_time_map:
                test_time_map[fname] = DEFAULT_TEST_DURATION

        time_keys = sorted(
            test_time_map.items(),
            key=lambda el: (el[1], el[0]),
            reverse=True)
    except FileNotFoundError:
        time_keys = [
            (os.path.normpath(file), DEFAULT_TEST_DURATION)
            for file in test_files
        ]

    def find_lowest_total_time(
            tests: list[tuple[float, list[str]]]) -> int:
        min_time: float | None = None
        res = None
        for ix, (e_time, _) in enumerate(tests):
            if min_time is None or e_time < min_time:
                min_time = e_time
                res = ix
        if res is None:
            raise ValueError("no nodes?")
        return res

    test_sets: list[tuple[float, list[str]]] = [
        (0.0, []) for _ in range(total_nodes)
    ]
    for key, timing in time_keys:
        ix = find_lowest_total_time(test_sets)
        min_time, min_arr = test_sets[ix]
        min_arr.append(key)
        test_sets[ix] = (min_time + timing, min_arr)
    print(f"{test_sets[cur_node][0]},{','.join(test_sets[cur_node][1])}")
