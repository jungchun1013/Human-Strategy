-- phpMyAdmin SQL Dump
-- version 4.9.0.1
-- https://www.phpmyadmin.net/
--
-- 主機： sql308.byetcluster.com
-- 產生時間： 2024 年 12 月 14 日 05:54
-- 伺服器版本： 10.6.19-MariaDB
-- PHP 版本： 7.2.22

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- 資料庫： `if0_37694833_human_strategy_exp_db`
--

-- --------------------------------------------------------

--
-- 資料表結構 `Results`
--

CREATE TABLE `Results` (
  `id` varchar(255) NOT NULL,
  `PHPSESSID` varchar(255) NOT NULL,
  `trial` varchar(255) NOT NULL,
  `attempt_num` int(11) NOT NULL,
  `tool` varchar(255) NOT NULL,
  `pos_x` float NOT NULL,
  `pos_y` float NOT NULL,
  `time` int(11) NOT NULL,
  `success_trial` tinyint(1) NOT NULL,
  `trial_order` bigint(64) NOT NULL,
  `success_place` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- 傾印資料表的資料 `Results`
--

INSERT INTO `Results` (`id`, `PHPSESSID`, `trial`, `attempt_num`, `tool`, `pos_x`, `pos_y`, `time`, `success_trial`, `trial_order`, `success_place`) VALUES
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Basic_1', 1, 'obj1', 174, 186, 9355, 1, 1734082016090, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Basic_1', 2, 'obj1', 203, 288, 36595, 1, 1734082016090, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Basic_1', 3, 'obj1', 186, 295, 44631, 1, 1734082016090, 1),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Basic_2', 1, 'obj2', 565, 465, 15283, 1, 1734082077278, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Basic_2', 2, 'obj1', 567, 439, 27579, 1, 1734082077278, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Basic_2', 3, 'obj2', 402, 236, 38571, 1, 1734082077278, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Basic_2', 4, 'obj3', 403, 233, 48395, 1, 1734082077278, 1),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Comp_GapCatapult', 1, 'obj2', 159, 411, 22505, 0, 1734082470670, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Comp_GapCatapult', 2, 'obj2', 487, 552, 32223, 0, 1734082470670, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Comp_GapCatapult', 3, 'obj2', 157, 415, 50418, 0, 1734082470670, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Comp_SoCloseLaunch', 1, 'obj2', 208, 250, 27430, 0, 1734082591225, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Comp_SoCloseLaunch', 2, 'obj3', 273, 331, 50278, 0, 1734082591225, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Comp_SoCloseLaunch', 3, 'obj1', 232, 274, 68514, 0, 1734082591225, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Comp_UnboxSlope', 1, 'obj1', 403, 327, 12604, 1, 1734082512902, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Comp_UnboxSlope', 2, 'obj2', 320, 569, 29340, 1, 1734082512902, 1),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Catapult_0', 1, 'obj2', 410, 558, 8697, 1, 1734082290622, 1),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Catapult_1', 1, 'obj1', 380, 551, 8053, 1, 1734082310861, 1),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Catapult_2', 1, 'obj2', 356, 291, 4255, 0, 1734082196858, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Catapult_2', 2, 'obj2', 353, 419, 11875, 0, 1734082196858, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Catapult_2', 3, 'obj2', 340, 295, 24039, 0, 1734082196858, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Catapult_3', 1, 'obj2', 244, 559, 10466, 1, 1734082233802, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Catapult_3', 2, 'obj2', 272, 562, 28982, 1, 1734082233802, 1),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Catapult_4', 1, 'obj1', 463, 273, 13348, 0, 1734082153666, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Catapult_4', 2, 'obj2', 502, 560, 22700, 0, 1734082153666, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Catapult_4', 3, 'obj3', 231, 478, 33640, 0, 1734082153666, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Gap_0', 1, 'obj1', 348, 152, 14955, 1, 1734082335158, 1),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Gap_1', 1, 'obj1', 320, 336, 11169, 0, 1734082388670, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Gap_1', 2, 'obj3', 327, 320, 21645, 0, 1734082388670, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Gap_1', 3, 'obj3', 324, 316, 34088, 0, 1734082388670, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Gap_2', 1, 'obj3', 286, 121, 17179, 1, 1734082108586, 1),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Gap_3', 1, 'obj1', 265, 39, 27651, 1, 1734082273318, 1),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Mech_Gap_4', 1, 'obj1', 323, 199, 8971, 1, 1734082409753, 1),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Playground', 1, 'obj3', 503, 458, 8091, 0, 1734081955969, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Playground', 2, 'obj3', 536, 327, 22732, 0, 1734081955969, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Playground', 3, 'obj2', 577, 373, 55432, 0, 1734081955969, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Playground', 4, 'obj2', 551, 404, 68103, 0, 1734081955969, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Playground', 5, 'obj2', 344, 66, 89716, 0, 1734081955969, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Playground', 6, 'obj3', 542, 121, 105552, 0, 1734081955969, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Playground', 7, 'obj3', 447, 105, 116560, 0, 1734081955969, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Playground', 8, 'obj1', 530, 152, 128384, 0, 1734081955969, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Playground', 9, 'obj1', 362, 286, 142137, 0, 1734081955969, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Playground', 10, 'obj1', 366, 126, 152633, 0, 1734081955969, 0),
('60fe7c857e049641f422a6f4', '937a62ef52e0bbe7b36ca204a8b86f10', 'Playground', 11, 'obj1', 368, 379, 175661, 0, 1734081955969, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Basic_1', 1, 'obj3', 141, 172, 52321, 1, 1734081495435, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Basic_1', 2, 'obj3', 548, 6.60001, 83200, 1, 1734081495435, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Basic_1', 3, 'obj3', 383, 171.6, 105474, 1, 1734081495435, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Basic_1', 4, 'obj3', 155, 171.6, 130372, 1, 1734081495435, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Basic_1', 5, 'obj1', 171, 180.6, 138745, 1, 1734081495435, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Basic_1', 6, 'obj2', 177, 188.6, 160482, 1, 1734081495435, 1),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Basic_2', 1, 'obj2', 401, 236.6, 25139, 1, 1734081247565, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Basic_2', 2, 'obj3', 402, 232.6, 34218, 1, 1734081247565, 1),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Comp_GapCatapult', 1, 'obj3', 285, 43, 48310, 0, 1734082793808, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Comp_GapCatapult', 2, 'obj3', 146, 440, 73863, 0, 1734082793808, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Comp_GapCatapult', 3, 'obj2', 490, 521, 88397, 0, 1734082793808, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Comp_SoCloseLaunch', 1, 'obj1', 259, 263, 33985, 1, 1734083057361, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Comp_SoCloseLaunch', 2, 'obj2', 262, 274, 76002, 1, 1734083057361, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Comp_SoCloseLaunch', 3, 'obj3', 259, 264, 102766, 1, 1734083057361, 1),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Comp_UnboxSlope', 1, 'obj1', 475, 328, 33266, 0, 1734082933905, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Comp_UnboxSlope', 2, 'obj1', 462, 326, 86618, 0, 1734082933905, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Comp_UnboxSlope', 3, 'obj2', 304, 412, 127977, 0, 1734082933905, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_0', 1, 'obj2', 463, 64, 15906, 0, 1734082691364, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_0', 2, 'obj3', 455, 110, 43017, 0, 1734082691364, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_0', 3, 'obj2', 460, 136, 52115, 0, 1734082691364, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_1', 1, 'obj1', 465, 147, 42636, 0, 1734082193290, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_1', 2, 'obj3', 414, 207, 95901, 0, 1734082193290, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_1', 3, 'obj1', 412, 231.6, 122004, 0, 1734082193290, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_2', 1, 'obj1', 311, 121.6, 65375, 0, 1734082410211, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_2', 2, 'obj3', 54, 135.6, 96388, 0, 1734082410211, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_2', 3, 'obj3', 342, 72.6, 132235, 0, 1734082410211, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_3', 1, 'obj2', 214, 207.6, 119405, 0, 1734081732211, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_3', 2, 'obj3', 227, 278.6, 148680, 0, 1734081732211, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_3', 3, 'obj1', 265, 287.6, 203511, 0, 1734081732211, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_4', 1, 'obj1', 203, 119, 32195, 0, 1734082482718, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_4', 2, 'obj3', 199, 162, 48445, 0, 1734082482718, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Catapult_4', 3, 'obj2', 225, 86.6, 66019, 0, 1734082482718, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Gap_0', 1, 'obj1', 349, 147, 38126, 1, 1734082061398, 1),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Gap_1', 1, 'obj1', 317, 432, 40073, 1, 1734082261389, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Gap_1', 2, 'obj3', 332, 444, 56729, 1, 1734082261389, 1),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Gap_2', 1, 'obj2', 299, 200, 221120, 1, 1734082010850, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Gap_2', 2, 'obj3', 242, 121, 265905, 1, 1734082010850, 1),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Gap_3', 1, 'obj1', 263, 64, 10808, 0, 1734082628289, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Gap_3', 2, 'obj1', 259, 140, 72495, 0, 1734082628289, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Gap_3', 3, 'obj2', 257, 53, 97280, 0, 1734082628289, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Mech_Gap_4', 1, 'obj1', 328, 228, 26689, 1, 1734082523147, 1),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Playground', 1, 'obj3', 460, 83, 3558, 0, 1734081187179, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Playground', 2, 'obj1', 410, 419.6, 40363, 0, 1734081187179, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Playground', 3, 'obj2', 519, 64, 65195, 0, 1734081187179, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Playground', 4, 'obj2', 370, 66, 86275, 0, 1734081187179, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Playground', 5, 'obj3', 443, 442, 104325, 0, 1734081187179, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Playground', 6, 'obj1', 518, 538, 118843, 0, 1734081187179, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Playground', 7, 'obj1', 399, 152, 141005, 0, 1734081187179, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Playground', 8, 'obj3', 449, 82, 151765, 0, 1734081187179, 0),
('615e80a2300de12db1a74cf4', '82a0ed8dca5e971a5fc14276dabaf033', 'Playground', 9, 'obj3', 397, 80, 172613, 0, 1734081187179, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Basic_1', 1, 'obj2', 190, 218, 7263, 1, 1734081097468, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Basic_2', 1, 'obj3', 402, 233, 9872, 1, 1734081065498, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Comp_GapCatapult', 1, 'obj2', 151, 423, 9185, 1, 1734081396191, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Comp_SoCloseLaunch', 1, 'obj3', 217, 257, 20832, 0, 1734081458130, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Comp_SoCloseLaunch', 2, 'obj2', 216, 248, 37742, 0, 1734081458130, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Comp_SoCloseLaunch', 3, 'obj1', 229, 270, 51612, 0, 1734081458130, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Comp_UnboxSlope', 1, 'obj2', 316, 531, 16829, 1, 1734081487587, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Catapult_0', 1, 'obj2', 399, 556, 8563, 1, 1734081265372, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Catapult_1', 1, 'obj1', 404, 554, 8442, 1, 1734081248734, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Catapult_2', 1, 'obj2', 320, 561, 7919, 1, 1734081351749, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Catapult_3', 1, 'obj2', 265, 561, 5578, 1, 1734081364227, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Catapult_4', 1, 'obj2', 268, 562, 8437, 1, 1734081232979, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Catapult_4', 2, 'obj2', 236, 563, 21191, 1, 1734081232979, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Catapult_4', 3, 'obj2', 280, 563, 38339, 1, 1734081232979, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Gap_0', 1, 'obj2', 345, 184, 3756, 1, 1734081335788, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Gap_0', 2, 'obj2', 352, 146, 14288, 1, 1734081335788, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Gap_1', 1, 'obj3', 323, 321, 8771, 1, 1734081312948, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Gap_1', 2, 'obj3', 318, 361, 19334, 1, 1734081312948, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Gap_2', 1, 'obj3', 251, 122, 9299, 1, 1734081284359, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Gap_3', 1, 'obj1', 288, 49, 11034, 0, 1734081160740, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Gap_3', 2, 'obj3', 299, 74, 33299, 0, 1734081160740, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Gap_3', 3, 'obj2', 238, 54, 43519, 0, 1734081160740, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Mech_Gap_4', 1, 'obj1', 325, 199, 7173, 1, 1734081187299, 1),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Playground', 1, 'obj1', 514, 162, 4199, 0, 1734081035587, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Playground', 2, 'obj3', 542, 313, 21622, 0, 1734081035587, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Playground', 3, 'obj3', 475, 492, 30116, 0, 1734081035587, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Playground', 4, 'obj3', 482, 478, 39186, 0, 1734081035587, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Playground', 5, 'obj1', 514, 546, 50557, 0, 1734081035587, 0),
('664f7b48d963b2432cf7139e', '8869016e68e8baf703a314457b044f06', 'Playground', 6, 'obj3', 554, 352, 63274, 0, 1734081035587, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Basic_1', 1, 'obj1', 32, 197, 30147, 1, 1734080998201, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Basic_1', 2, 'obj3', 144, 179, 38571, 1, 1734080998201, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Basic_1', 3, 'obj2', 177, 189, 54276, 1, 1734080998201, 1),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Basic_2', 1, 'obj3', 191, 124, 13137, 1, 1734080883437, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Basic_2', 2, 'obj3', 399, 236, 24066, 1, 1734080883437, 1),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Catapult_1', 1, 'obj1', 96, 223, 15151, 0, 1734081410138, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Catapult_1', 2, 'obj2', 104, 193, 25375, 0, 1734081410138, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Catapult_1', 3, 'obj1', 226, 109, 35991, 0, 1734081410138, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Catapult_3', 1, 'obj2', 200, 236, 8293, 0, 1734081326422, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Catapult_3', 2, 'obj2', 205, 249, 23118, 0, 1734081326422, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Catapult_3', 3, 'obj2', 556, 255, 32230, 0, 1734081326422, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Catapult_4', 1, 'obj3', 269, 163, 7696, 0, 1734081368121, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Catapult_4', 2, 'obj2', 269, 180, 19672, 0, 1734081368121, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Catapult_4', 3, 'obj2', 235, 180, 31176, 0, 1734081368121, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Gap_0', 1, 'obj1', 43, 465, 7415, 1, 1734081283454, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Gap_0', 2, 'obj2', 346, 187, 33024, 1, 1734081283454, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Gap_0', 3, 'obj2', 337, 170, 44200, 1, 1734081283454, 1),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Gap_1', 1, 'obj2', 29, 567, 9706, 0, 1734081449755, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Gap_1', 2, 'obj3', 34, 549, 21970, 0, 1734081449755, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Gap_1', 3, 'obj1', 64, 567, 32099, 0, 1734081449755, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Gap_2', 1, 'obj3', 245, 153, 18494, 1, 1734081229851, 1),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Gap_4', 1, 'obj3', 555, 327, 21758, 0, 1734081201826, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Gap_4', 2, 'obj1', 377, 224, 108569, 0, 1734081201826, 0),
('66631cb0def915d06af2b88b', '435424a19d0ae8be780f42f8d492d65a', 'Mech_Gap_4', 3, 'obj3', 329, 214, 133458, 0, 1734081201826, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Basic_1', 1, 'obj2', 441.6, 149.4, 14454, 1, 1734081129512, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Basic_1', 2, 'obj2', 185.6, 251.4, 21971, 1, 1734081129512, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Basic_2', 1, 'obj2', 407.6, 278.4, 14695, 1, 1734081091352, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Basic_2', 2, 'obj1', 403.6, 238.4, 22324, 1, 1734081091352, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Basic_2', 3, 'obj3', 400.6, 266.4, 33019, 1, 1734081091352, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Comp_GapCatapult', 1, 'obj2', 148.6, 435.4, 12034, 1, 1734081573574, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Comp_SoCloseLaunch', 1, 'obj2', 260.6, 278.4, 27479, 1, 1734081550654, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Comp_SoCloseLaunch', 2, 'obj3', 253.6, 289.4, 48445, 1, 1734081550654, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Comp_UnboxSlope', 1, 'obj2', 325.6, 559.4, 15221, 1, 1734081486052, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Catapult_0', 1, 'obj2', 422.6, 560, 7748, 1, 1734081326874, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Catapult_1', 1, 'obj1', 425.6, 541.4, 9357, 1, 1734081366802, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Catapult_2', 1, 'obj2', 313.6, 438.4, 8085, 1, 1734081162690, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Catapult_2', 2, 'obj2', 331.6, 555.4, 18285, 1, 1734081162690, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Catapult_3', 1, 'obj2', 257.6, 558.4, 5477, 1, 1734081454054, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Catapult_4', 1, 'obj3', 236.6, 568.4, 5751, 0, 1734081267804, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Catapult_4', 2, 'obj3', 231.6, 574.4, 19467, 0, 1734081267804, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Catapult_4', 3, 'obj2', 205.6, 558.4, 29609, 0, 1734081267804, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Gap_0', 1, 'obj1', 299.6, 394.4, 18103, 1, 1734081218998, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Gap_0', 2, 'obj1', 341.6, 339.4, 47648, 1, 1734081218998, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Gap_1', 1, 'obj1', 322.6, 354.4, 11685, 1, 1734081441202, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Gap_1', 2, 'obj3', 320.6, 358.4, 25001, 1, 1734081441202, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Gap_2', 1, 'obj3', 259.6, 170.4, 15914, 1, 1734081351010, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Gap_3', 1, 'obj1', 264.6, 54.4, 15501, 1, 1734081408054, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Gap_3', 2, 'obj1', 265.6, 62.4, 31288, 1, 1734081408054, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Gap_4', 1, 'obj3', 329.6, 214.4, 20436, 1, 1734081311130, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Mech_Gap_4', 2, 'obj1', 325.6, 226.4, 32905, 1, 1734081311130, 1),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Playground', 1, 'obj3', 232.6, 187.4, 11300, 0, 1734081037529, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Playground', 2, 'obj3', 169.6, 490.4, 19111, 0, 1734081037529, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Playground', 3, 'obj1', 414.6, 457.4, 39102, 0, 1734081037529, 0),
('66d4d4c9d991eff770dec514', '63578d800ade8edc247e1a2eb54eeb3a', 'Playground', 4, 'obj3', 514.6, 556.4, 64563, 0, 1734081037529, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Basic_1', 1, 'obj3', 152, 208, 21287, 1, 1734081782841, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Basic_1', 2, 'obj1', 194, 222, 54442, 1, 1734081782841, 1),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Basic_2', 1, 'obj2', 566, 374, 18040, 1, 1734081948223, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Basic_2', 2, 'obj3', 562, 369, 35607, 1, 1734081948223, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Basic_2', 3, 'obj1', 567, 360, 85274, 1, 1734081948223, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Basic_2', 4, 'obj1', 564, 573, 99355, 1, 1734081948223, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Basic_2', 5, 'obj2', 394, 525, 114034, 1, 1734081948223, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Basic_2', 6, 'obj3', 402, 462, 127260, 1, 1734081948223, 1),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Comp_GapCatapult', 1, 'obj2', 161, 575, 20807, 0, 1734082986030, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Comp_GapCatapult', 2, 'obj1', 180, 565, 33925, 0, 1734082986030, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Comp_GapCatapult', 3, 'obj1', 140, 566, 45941, 0, 1734082986030, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Comp_SoCloseLaunch', 1, 'obj1', 502, 558, 15188, 0, 1734083179065, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Comp_SoCloseLaunch', 2, 'obj1', 250, 566, 41510, 0, 1734083179065, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Comp_SoCloseLaunch', 3, 'obj2', 228, 514, 62755, 0, 1734083179065, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Comp_UnboxSlope', 1, 'obj2', 343, 575, 55718, 1, 1734083085042, 1),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Catapult_0', 1, 'obj2', 440, 529, 12021, 1, 1734082547394, 1),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Catapult_1', 1, 'obj1', 406, 533, 12213, 1, 1734082890575, 1),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Catapult_2', 1, 'obj2', 324, 218, 11168, 0, 1734082059483, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Catapult_2', 2, 'obj1', 315, 462, 29050, 0, 1734082059483, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Catapult_2', 3, 'obj3', 290, 421, 58304, 0, 1734082059483, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Catapult_3', 1, 'obj2', 257, 446, 10143, 0, 1734082137943, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Catapult_3', 2, 'obj2', 239, 492, 34973, 0, 1734082137943, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Catapult_3', 3, 'obj2', 224, 501, 62899, 0, 1734082137943, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Catapult_4', 1, 'obj3', 241, 541, 22774, 1, 1734082598240, 1),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Gap_0', 1, 'obj1', 350, 552, 12412, 1, 1734082244111, 1),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Gap_1', 1, 'obj1', 308, 503, 9567, 0, 1734082447975, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Gap_1', 2, 'obj1', 308, 587, 23743, 0, 1734082447975, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Gap_1', 3, 'obj3', 317, 569, 180084, 0, 1734082447975, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Gap_2', 1, 'obj3', 260, 516, 17362, 1, 1734082716934, 1),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Gap_3', 1, 'obj1', 265, 502, 12490, 0, 1734082673653, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Gap_3', 2, 'obj3', 240, 510, 43977, 0, 1734082673653, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Gap_3', 3, 'obj3', 273, 513, 59112, 0, 1734082673653, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Gap_4', 1, 'obj2', 345, 516, 20261, 0, 1734082860996, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Gap_4', 2, 'obj3', 351, 549, 37235, 0, 1734082860996, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Mech_Gap_4', 3, 'obj3', 361, 540, 69197, 0, 1734082860996, 0),
('675a8899cd7510c8cfba99e6@email.prolific.com', 'deb20c9a879d2cfb2a0c91a33059f13b', 'Playground', 1, 'obj1', 557, 99, 29422, 0, 1734081693748, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Basic_1', 1, 'obj1', 186, 233, 21310, 1, 1733439709402, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Basic_1', 2, 'obj1', 197, 569, 28058, 1, 1733439709402, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Basic_1', 3, 'obj2', 191, 340, 41179, 1, 1733439709402, 1),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Basic_2', 1, 'obj1', 51, 281, 8522, 1, 1733439651000, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Basic_2', 2, 'obj1', 57, 474, 15710, 1, 1733439651000, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Basic_2', 3, 'obj1', 48, 569, 32803, 1, 1733439651000, 1),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 1, 'obj3', 152, 504, 18483, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 2, 'obj1', 155, 424, 35111, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 3, 'obj2', 159, 426, 45147, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 4, 'obj2', 131, 425, 51698, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 5, 'obj1', 136, 424, 69322, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 6, 'obj1', 145, 426, 109566, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 7, 'obj2', 150, 499, 127150, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 8, 'obj2', 135, 424, 134665, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 9, 'obj2', 135, 490, 150667, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 10, 'obj2', 137, 486, 160427, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 11, 'obj2', 156, 489, 167207, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 12, 'obj2', 140, 510, 180572, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 13, 'obj1', 136, 427, 208948, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 14, 'obj1', 159, 507, 231005, 1, 1733440218144, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_GapCatapult', 15, 'obj2', 147, 423, 246697, 1, 1733440218144, 1),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_SoCloseLaunch', 1, 'obj3', 175, 341, 33248, 0, 1733439946846, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_SoCloseLaunch', 2, 'obj3', 201, 268, 45410, 0, 1733439946846, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_SoCloseLaunch', 3, 'obj3', 231, 265, 48698, 0, 1733439946846, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_SoCloseLaunch', 4, 'obj3', 223, 268, 53778, 0, 1733439946846, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Comp_UnboxSlope', 1, 'obj1', 349, 516, 7416, 1, 1733439870255, 1),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Mech_Launch_v2_1', 1, 'obj1', 125, 203, 4946, 1, 1733439828473, 0),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Mech_Launch_v2_1', 2, 'obj1', 137, 533, 14407, 1, 1733439828473, 1),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Mech_Launch_v2_2', 1, 'obj2', 529, 571, 13375, 1, 1733439759499, 1),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Mech_SoClose_1', 1, 'obj3', 469, 372, 5133, 1, 1733439800814, 1),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Mech_SoClose_2', 1, 'obj3', 165, 459.2, 6346, 1, 1733439777918, 1),
('eleanor', 'b0ef11d257943b06bbf6e9ac1070bd13', 'Playground', 1, 'obj1', 505, 257.6, 23071, 0, 1733439598853, 0);

--
-- 已傾印資料表的索引
--

--
-- 資料表索引 `Results`
--
ALTER TABLE `Results`
  ADD PRIMARY KEY (`id`,`trial`,`attempt_num`),
  ADD KEY `trial` (`trial`);

--
-- 已傾印資料表的限制式
--

--
-- 資料表的限制式 `Results`
--
ALTER TABLE `Results`
  ADD CONSTRAINT `results_ibfk_1` FOREIGN KEY (`id`) REFERENCES `Users` (`name`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
