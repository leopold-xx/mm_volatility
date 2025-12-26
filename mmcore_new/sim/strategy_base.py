# -*- coding: utf-8 -*-
"""
策略基类和接口定义
兼容numba加速的回测引擎
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class StrategyBase(ABC):
    """
    策略基类（抽象类）

    所有策略必须继承此类并实现相应的回调方法

    生命周期：
        1. on_start: 回测开始时调用一次
        2. on_trade: 每个trade数据到达时调用
        3. on_order_filled: 订单成交时调用
        4. on_stop: 回测结束时调用一次

    注意：
        - 策略的on_trade方法应该尽可能简单，避免复杂计算
        - 未来版本将支持numba加速的策略主循环
    """

    def __init__(self, name: str = "BaseStrategy"):
        """
        初始化策略

        参数：
            name: 策略名称
        """
        self.name = name
        self.is_running = False

        # 策略参数（子类可以扩展）
        self.params = {}

        # 策略状态（子类可以扩展）
        self.state = {}

        # 性能统计
        self.stats = {
            'orders_sent': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'trades_processed': 0,
        }

    @abstractmethod
    def on_start(self, engine: Any) -> None:
        """
        策略启动回调

        在回测开始时调用一次，用于初始化策略状态

        参数：
            engine: 回测引擎实例
        """
        self.is_running = True
        print(f"📊 策略 {self.name} 启动")

    @abstractmethod
    def on_trade(self, trade_data: Dict, engine: Any) -> None:
        """
        Trade数据回调（核心逻辑）

        每个trade到达时调用，策略的主要逻辑在这里实现

        参数：
            trade_data: trade数据字典，包含：
                - timestamp: 时间戳（毫秒）
                - trade_price: 成交价格
                - trade_qty: 成交数量
                - trade_side: 成交方向（'buy' 或 'sell'）
            engine: 回测引擎实例

        可以调用的引擎方法：
            - engine.place_order(price, quantity, side): 发送订单
            - engine.cancel_order(order_id): 撤销订单
            - engine.get_account_status(): 获取账户状态
            - engine.get_order_status(order_id): 获取订单状态

        注意：
            - 此方法会被频繁调用，应保持逻辑简单
            - 避免复杂的计算和I/O操作
        """
        self.stats['trades_processed'] += 1

    def on_order_filled(self, order_id: int, fill_price: float,
                       fill_qty: float, engine: Any) -> None:
        """
        订单成交回调

        订单成交时调用

        参数：
            order_id: 订单ID
            fill_price: 成交价格
            fill_qty: 成交数量
            engine: 回测引擎实例
        """
        self.stats['orders_filled'] += 1

    @abstractmethod
    def on_stop(self, engine: Any) -> None:
        """
        策略停止回调

        在回测结束时调用一次，用于清理和总结

        参数：
            engine: 回测引擎实例
        """
        self.is_running = False
        print(f"📊 策略 {self.name} 停止")
        print(f"   处理trades: {self.stats['trades_processed']}")
        print(f"   发送订单: {self.stats['orders_sent']}")
        print(f"   成交订单: {self.stats['orders_filled']}")

    def get_stats(self) -> Dict:
        """
        获取策略统计信息

        返回：
            策略统计字典
        """
        return self.stats.copy()

    def reset(self) -> None:
        """
        重置策略状态

        用于多次回测时重置策略
        """
        self.is_running = False
        self.state = {}
        self.stats = {
            'orders_sent': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'trades_processed': 0,
        }


class SimpleMarketMaker(StrategyBase):
    """
    简单做市策略

    策略逻辑：
        1. 在最新成交价两侧挂单
        2. 买单价格 = 最新价 * (1 - spread/2)
        3. 卖单价格 = 最新价 * (1 + spread/2)
        4. 成交后立即补单
        5. 限制最大持仓
    """

    def __init__(self,
                 spread: float = 0.001,
                 order_size: float = 0.01,
                 max_position: float = 1.0,
                 update_threshold: float = 0.0005,
                 name: str = "SimpleMarketMaker"):
        """
        初始化简单做市策略

        参数：
            spread: 价差比例（默认0.1%）
            order_size: 每笔订单大小
            max_position: 最大持仓限制（绝对值）
            update_threshold: 价格更新阈值（相对变化）
            name: 策略名称
        """
        super().__init__(name)

        # 策略参数
        self.params = {
            'spread': spread,
            'order_size': order_size,
            'max_position': max_position,
            'update_threshold': update_threshold,
        }

        # 策略状态
        self.state = {
            'active_buy_order': None,
            'active_sell_order': None,
            'last_mid_price': 0.0,
            'position': 0.0,
            'pending_cancels': set(),  # 待撤销的订单集合
        }

    def on_start(self, engine: Any) -> None:
        """策略启动"""
        super().on_start(engine)
        print(f"   价差: {self.params['spread']*100:.2f}%")
        print(f"   订单大小: {self.params['order_size']}")
        print(f"   最大持仓: {self.params['max_position']}")
        print(f"   更新阈值: {self.params['update_threshold']*100:.2f}%")

    def on_trade(self, trade_data: Dict, engine: Any) -> None:
        """
        处理trade数据，更新挂单
        """
        super().on_trade(trade_data, engine)

        # 获取最新价格
        current_price = trade_data['trade_price']

        # 第一次运行，初始化
        if self.state['last_mid_price'] == 0:
            self.state['last_mid_price'] = current_price

        # 获取账户状态
        account = engine.get_account_status()
        self.state['position'] = account['position']

        # 检查价格变化是否需要更新订单
        price_change = abs(current_price - self.state['last_mid_price']) / self.state['last_mid_price']
        should_update = price_change > self.params['update_threshold']

        # 撤销旧订单（如果价格变化较大）
        if should_update:
            self._cancel_active_orders(engine)
            self.state['last_mid_price'] = current_price

        # 发送新订单
        self._place_new_orders(current_price, engine)

    def _cancel_active_orders(self, engine: Any) -> None:
        """撤销活跃订单"""
        # 撤销买单
        if (self.state['active_buy_order'] is not None and
            self.state['active_buy_order'] not in self.state['pending_cancels']):

            success = engine.cancel_order(self.state['active_buy_order'])
            if success:
                self.state['pending_cancels'].add(self.state['active_buy_order'])
            self.state['active_buy_order'] = None

        # 撤销卖单
        if (self.state['active_sell_order'] is not None and
            self.state['active_sell_order'] not in self.state['pending_cancels']):

            success = engine.cancel_order(self.state['active_sell_order'])
            if success:
                self.state['pending_cancels'].add(self.state['active_sell_order'])
            self.state['active_sell_order'] = None

    def _place_new_orders(self, current_price: float, engine: Any) -> None:
        """发送新订单"""
        # 计算挂单价格
        half_spread = self.params['spread'] / 2
        buy_price = current_price * (1 - half_spread)
        sell_price = current_price * (1 + half_spread)

        # 发送买单（如果没有活跃买单且持仓未超限）
        if (self.state['active_buy_order'] is None and
            self.state['position'] < self.params['max_position']):

            order_id = engine.place_order(
                price=buy_price,
                quantity=self.params['order_size'],
                side='buy'
            )

            if order_id > 0:
                self.state['active_buy_order'] = order_id
                self.stats['orders_sent'] += 1

        # 发送卖单（如果没有活跃卖单且持仓允许）
        if (self.state['active_sell_order'] is None and
            self.state['position'] > -self.params['max_position']):

            order_id = engine.place_order(
                price=sell_price,
                quantity=self.params['order_size'],
                side='sell'
            )

            if order_id > 0:
                self.state['active_sell_order'] = order_id
                self.stats['orders_sent'] += 1

    def on_order_filled(self, order_id: int, fill_price: float,
                       fill_qty: float, engine: Any) -> None:
        """
        订单成交回调，清除对应的活跃订单记录
        """
        super().on_order_filled(order_id, fill_price, fill_qty, engine)

        # 从pending_cancels中移除（如果存在）
        self.state['pending_cancels'].discard(order_id)

        # 清除活跃订单记录
        if order_id == self.state['active_buy_order']:
            self.state['active_buy_order'] = None

        elif order_id == self.state['active_sell_order']:
            self.state['active_sell_order'] = None

    def on_stop(self, engine: Any) -> None:
        """策略停止，输出最终统计"""
        super().on_stop(engine)

        # 获取最终账户状态
        account = engine.get_account_status()

        print(f"\n📈 策略最终状态:")
        print(f"   最终持仓: {account['position']:.4f}")
        print(f"   最终资金: {account['cash']:.2f}")
        print(f"   总价值: {account['total_value']:.2f}")

        if self.stats['orders_sent'] > 0:
            fill_rate = self.stats['orders_filled'] / self.stats['orders_sent'] * 100
            print(f"   成交率: {fill_rate:.1f}%")


class DummyStrategy(StrategyBase):
    """
    空策略（用于测试）

    不进行任何交易，只记录市场数据
    """

    def __init__(self, name: str = "DummyStrategy"):
        super().__init__(name)
        self.price_history = []
        self.trade_count = 0

    def on_start(self, engine: Any) -> None:
        """策略启动"""
        super().on_start(engine)
        print("   模式: 只观察，不交易")

    def on_trade(self, trade_data: Dict, engine: Any) -> None:
        """记录价格"""
        super().on_trade(trade_data, engine)

        self.price_history.append(trade_data['trade_price'])
        self.trade_count += 1

        # 每10000个trade输出一次
        if self.trade_count % 10000 == 0:
            avg_price = sum(self.price_history[-1000:]) / min(1000, len(self.price_history))
            print(f"   已处理 {self.trade_count} 个trades, 最近均价: {avg_price:.2f}")

    def on_stop(self, engine: Any) -> None:
        """策略停止"""
        super().on_stop(engine)

        if self.price_history:
            print(f"\n📊 市场数据统计:")
            print(f"   总trades: {self.trade_count}")
            print(f"   最低价: {min(self.price_history):.2f}")
            print(f"   最高价: {max(self.price_history):.2f}")
            print(f"   均价: {sum(self.price_history)/len(self.price_history):.2f}")


class AggressiveMarketMaker(StrategyBase):
    """
    激进做市策略

    特点：
        1. 更小的价差
        2. 更频繁的订单更新
        3. 更大的订单数量
    """

    def __init__(self,
                 spread: float = 0.0005,
                 order_size: float = 0.02,
                 max_position: float = 0.5,
                 update_threshold: float = 0.0002,
                 name: str = "AggressiveMarketMaker"):
        super().__init__(name)

        self.params = {
            'spread': spread,
            'order_size': order_size,
            'max_position': max_position,
            'update_threshold': update_threshold,
        }

        self.state = {
            'active_orders': {},  # order_id -> side
            'last_price': 0.0,
            'position': 0.0,
        }

    def on_start(self, engine: Any) -> None:
        super().on_start(engine)
        print(f"   激进参数 - 价差: {self.params['spread']*100:.3f}%")

    def on_trade(self, trade_data: Dict, engine: Any) -> None:
        super().on_trade(trade_data, engine)

        current_price = trade_data['trade_price']

        # 更新持仓
        account = engine.get_account_status()
        self.state['position'] = account['position']

        # 检查是否需要更新
        if self.state['last_price'] == 0:
            should_update = True
        else:
            price_change = abs(current_price - self.state['last_price']) / self.state['last_price']
            should_update = price_change > self.params['update_threshold']

        if should_update:
            # 撤销所有活跃订单
            for order_id in list(self.state['active_orders'].keys()):
                engine.cancel_order(order_id)

            self.state['active_orders'].clear()

            # 发送新订单
            half_spread = self.params['spread'] / 2
            buy_price = current_price * (1 - half_spread)
            sell_price = current_price * (1 + half_spread)

            # 买单
            if abs(self.state['position']) < self.params['max_position']:
                buy_order_id = engine.place_order(buy_price, self.params['order_size'], 'buy')
                if buy_order_id > 0:
                    self.state['active_orders'][buy_order_id] = 'buy'
                    self.stats['orders_sent'] += 1

            # 卖单
            if abs(self.state['position']) < self.params['max_position']:
                sell_order_id = engine.place_order(sell_price, self.params['order_size'], 'sell')
                if sell_order_id > 0:
                    self.state['active_orders'][sell_order_id] = 'sell'
                    self.stats['orders_sent'] += 1

            self.state['last_price'] = current_price

    def on_order_filled(self, order_id: int, fill_price: float,
                       fill_qty: float, engine: Any) -> None:
        super().on_order_filled(order_id, fill_price, fill_qty, engine)
        # 从活跃订单中移除
        self.state['active_orders'].pop(order_id, None)

    def on_stop(self, engine: Any) -> None:
        super().on_stop(engine)
        account = engine.get_account_status()
        print(f"\n📈 激进策略最终状态:")
        print(f"   最终持仓: {account['position']:.4f}")
        print(f"   总价值: {account['total_value']:.2f}")