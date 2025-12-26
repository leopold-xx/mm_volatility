# -*- coding: utf-8 -*-
"""
数据扫描和日期检查工具
扫描指定路径下的数据文件，检查可用日期范围
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class DataScanner:
    """数据扫描器 - 扫描和管理交易数据文件"""

    def __init__(self, base_path: str, data_pattern: str):
        """
        初始化数据扫描器

        参数:
            base_path: 数据根路径
            data_pattern: 数据文件名模式，支持{symbol}、{symbol_lower}、{date}占位符
                例如: '{symbol}-aggTrades-{date}.zip'
                例如: 'binance_usd_{symbol_lower}/aggtrades/{symbol}-aggTrades-{date}.zip'
        """
        self.base_path = Path(base_path)
        self.data_pattern = data_pattern

        # 编译正则表达式用于匹配日期
        self.date_regex = re.compile(r'(\d{4}-\d{2}-\d{2})')

    def scan_symbol_dates(self, symbol: str) -> Set[str]:
        """
        扫描指定品种的所有可用日期

        参数:
            symbol: 交易品种（如'BTCUSDT'）

        返回:
            可用日期集合（YYYY-MM-DD格式）
        """
        available_dates = set()

        try:
            # 替换路径模式中的占位符
            pattern_path = self.data_pattern.format(
                symbol=symbol,
                symbol_lower=symbol.lower(),
                date='*'  # 使用通配符匹配所有日期
            )

            # 构建完整搜索路径
            search_path = self.base_path / pattern_path.replace('*', '')
            search_dir = search_path.parent

            if not search_dir.exists():
                logger.warning(f"数据目录不存在: {search_dir}")
                return available_dates

            # 搜索匹配的文件
            pattern_filename = search_path.name.replace('*', '')

            # 如果模式中包含日期占位符，需要特殊处理
            if '{date}' in pattern_filename:
                # 构建文件名匹配模式
                filename_pattern = pattern_filename.replace('{date}', r'(\d{4}-\d{2}-\d{2})')
                filename_regex = re.compile(filename_pattern)

                # 扫描目录中的所有文件
                for file_path in search_dir.iterdir():
                    if file_path.is_file():
                        match = filename_regex.search(file_path.name)
                        if match:
                            date_str = match.group(1)
                            # 验证日期格式
                            try:
                                datetime.strptime(date_str, '%Y-%m-%d')
                                available_dates.add(date_str)
                            except ValueError:
                                continue
            else:
                # 如果文件名中没有日期占位符，从文件名中提取日期
                for file_path in search_dir.iterdir():
                    if file_path.is_file() and symbol in file_path.name:
                        match = self.date_regex.search(file_path.name)
                        if match:
                            date_str = match.group(1)
                            available_dates.add(date_str)

        except Exception as e:
            logger.error(f"扫描品种 {symbol} 数据时出错: {e}")

        return available_dates

    def get_file_path(self, symbol: str, date: str) -> Path:
        """
        获取指定品种和日期的数据文件路径

        参数:
            symbol: 交易品种
            date: 日期（YYYY-MM-DD格式）

        返回:
            数据文件的完整路径
        """
        # 替换路径模式中的占位符
        file_path = self.data_pattern.format(
            symbol=symbol,
            symbol_lower=symbol.lower(),
            date=date
        )

        return self.base_path / file_path

    def check_file_exists(self, symbol: str, date: str) -> bool:
        """检查指定品种和日期的数据文件是否存在"""
        file_path = self.get_file_path(symbol, date)
        return file_path.exists() and file_path.is_file()

    def generate_date_range(self, start_date: str, end_date: str,
                          skip_weekends: bool = False) -> List[str]:
        """
        生成日期范围内的所有日期

        参数:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            skip_weekends: 是否跳过周末

        返回:
            日期字符串列表
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        dates = []
        current = start

        while current <= end:
            # 如果需要跳过周末（0=周一，6=周日）
            if skip_weekends and current.weekday() in [5, 6]:
                current += timedelta(days=1)
                continue

            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        return dates

    def scan_all_symbols(self, symbols: List[str],
                        start_date: str = None, end_date: str = None,
                        skip_weekends: bool = False) -> Dict[str, Dict]:
        """
        扫描所有品种的数据可用性

        参数:
            symbols: 品种列表
            start_date: 可选的开始日期过滤
            end_date: 可选的结束日期过滤
            skip_weekends: 是否跳过周末

        返回:
            品种数据信息字典
        """
        result = {}

        # 生成目标日期范围（如果指定了）
        target_dates = None
        if start_date and end_date:
            target_dates = set(self.generate_date_range(
                start_date, end_date, skip_weekends
            ))

        for symbol in symbols:
            logger.info(f"扫描品种: {symbol}")

            # 获取可用日期
            available_dates = self.scan_symbol_dates(symbol)

            # 如果指定了日期范围，则过滤
            if target_dates:
                available_dates = available_dates.intersection(target_dates)
                missing_dates = target_dates - available_dates
            else:
                missing_dates = set()

            # 统计信息
            symbol_info = {
                'available_dates': sorted(list(available_dates)),
                'missing_dates': sorted(list(missing_dates)),
                'total_available': len(available_dates),
                'total_missing': len(missing_dates),
                'coverage_rate': len(available_dates) / len(target_dates) if target_dates else 1.0,
                'date_range': {
                    'first_date': min(available_dates) if available_dates else None,
                    'last_date': max(available_dates) if available_dates else None
                }
            }

            result[symbol] = symbol_info

        return result

    def filter_valid_combinations(self, symbols: List[str], dates: List[str]) -> List[Tuple[str, str]]:
        """
        过滤出有效的(品种, 日期)组合

        参数:
            symbols: 品种列表
            dates: 日期列表

        返回:
            有效组合的列表 [(symbol, date), ...]
        """
        valid_combinations = []

        for symbol in symbols:
            available_dates = self.scan_symbol_dates(symbol)
            for date in dates:
                if date in available_dates:
                    valid_combinations.append((symbol, date))

        return valid_combinations

    def get_data_summary(self, symbols: List[str],
                        start_date: str = None, end_date: str = None) -> Dict:
        """
        获取数据概况摘要

        返回:
            数据摘要统计
        """
        scan_result = self.scan_all_symbols(symbols, start_date, end_date)

        # 计算总体统计
        total_files = sum(info['total_available'] for info in scan_result.values())
        total_missing = sum(info['total_missing'] for info in scan_result.values())

        symbols_with_data = sum(1 for info in scan_result.values()
                               if info['total_available'] > 0)

        # 计算平均覆盖率
        avg_coverage = sum(info['coverage_rate'] for info in scan_result.values()) / len(symbols) if symbols else 0

        # 找出数据最完整和最不完整的品种
        best_symbol = max(scan_result.items(),
                         key=lambda x: x[1]['coverage_rate']) if scan_result else None
        worst_symbol = min(scan_result.items(),
                          key=lambda x: x[1]['coverage_rate']) if scan_result else None

        summary = {
            'total_symbols': len(symbols),
            'symbols_with_data': symbols_with_data,
            'total_files_found': total_files,
            'total_files_missing': total_missing,
            'average_coverage_rate': avg_coverage,
            'best_coverage': {
                'symbol': best_symbol[0] if best_symbol else None,
                'rate': best_symbol[1]['coverage_rate'] if best_symbol else 0
            },
            'worst_coverage': {
                'symbol': worst_symbol[0] if worst_symbol else None,
                'rate': worst_symbol[1]['coverage_rate'] if worst_symbol else 0
            },
            'date_range_analyzed': f"{start_date} ~ {end_date}" if start_date and end_date else "全部",
            'symbol_details': scan_result
        }

        return summary


def print_data_summary(summary: Dict):
    """打印数据摘要报告"""
    print("📊 数据扫描摘要报告")
    print("=" * 60)
    print(f"品种总数: {summary['total_symbols']}")
    print(f"有数据品种: {summary['symbols_with_data']}")
    print(f"找到文件数: {summary['total_files_found']}")
    print(f"缺失文件数: {summary['total_files_missing']}")
    print(f"平均覆盖率: {summary['average_coverage_rate']:.2%}")
    print(f"分析日期范围: {summary['date_range_analyzed']}")

    if summary['best_coverage']['symbol']:
        print(f"\n📈 最佳覆盖: {summary['best_coverage']['symbol']} ({summary['best_coverage']['rate']:.2%})")
    if summary['worst_coverage']['symbol']:
        print(f"📉 最差覆盖: {summary['worst_coverage']['symbol']} ({summary['worst_coverage']['rate']:.2%})")

    print(f"\n📋 品种详情:")
    for symbol, info in summary['symbol_details'].items():
        print(f"  {symbol:12}: {info['total_available']:3}个文件 "
              f"({info['coverage_rate']:.1%} 覆盖率)")
        if info['missing_dates']:
            missing_preview = info['missing_dates'][:3]
            if len(info['missing_dates']) > 3:
                missing_preview.append(f"... (+{len(info['missing_dates'])-3})")
            print(f"              缺失: {', '.join(missing_preview)}")


if __name__ == "__main__":
    # 测试数据扫描器
    from batch_config import ACTIVE_DATA_CONFIG, ACTIVE_SYMBOLS, DATE_RANGE_CONFIG

    scanner = DataScanner(
        base_path=ACTIVE_DATA_CONFIG['base_path'],
        data_pattern=ACTIVE_DATA_CONFIG['data_pattern']
    )

    # 扫描数据
    summary = scanner.get_data_summary(
        symbols=ACTIVE_SYMBOLS,
        start_date=DATE_RANGE_CONFIG['start_date'],
        end_date=DATE_RANGE_CONFIG['end_date']
    )

    # 打印报告
    print_data_summary(summary)