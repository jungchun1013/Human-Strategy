-- phpMyAdmin SQL Dump
-- version 4.9.0.1
-- https://www.phpmyadmin.net/
--
-- 主機： sql308.byetcluster.com
-- 產生時間： 2024 年 12 月 18 日 13:15
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
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Basic_1', 1, 'obj2', 440.6, 104.4, 7360, 1, 1734543879354, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Basic_1', 2, 'obj3', 389.6, 171.4, 41233, 1, 1734543879354, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Basic_1', 3, 'obj1', 412.6, 180.4, 56869, 1, 1734543879354, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Basic_1', 4, 'obj3', 394.6, 203.4, 69934, 1, 1734543879354, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Basic_1', 5, 'obj1', 407.6, 183.4, 76655, 1, 1734543879354, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Basic_1', 6, 'obj3', 553.6, 52.4, 83020, 1, 1734543879354, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Basic_1', 7, 'obj2', 190.6, 217.4, 97239, 1, 1734543879354, 1),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Basic_2', 1, 'obj2', 400.6, 238.4, 4633, 1, 1734543761725, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Basic_2', 2, 'obj3', 403.6, 232.4, 22952, 1, 1734543761725, 1),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Comp_GapCatapult', 1, 'obj1', 138.6, 422.4, 28884, 1, 1734544482268, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Comp_GapCatapult', 2, 'obj3', 151.6, 435.4, 59798, 1, 1734544482268, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Comp_GapCatapult', 3, 'obj2', 150.6, 419.4, 76896, 1, 1734544482268, 1),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Comp_SoCloseLaunch', 1, 'obj3', 271.6, 296.4, 16060, 1, 1734544392564, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Comp_SoCloseLaunch', 2, 'obj3', 265.6, 268.4, 34863, 1, 1734544392564, 1),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Comp_UnboxSlope', 1, 'obj1', 452.6, 322.4, 17451, 0, 1734544598041, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Comp_UnboxSlope', 2, 'obj1', 361.6, 324.4, 59526, 0, 1734544598041, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Comp_UnboxSlope', 3, 'obj1', 284.6, 433.4, 98200, 0, 1734544598041, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_Launch_v2_0', 1, 'obj1', 81.6, 270.4, 9321, 0, 1734544179492, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_Launch_v2_0', 2, 'obj3', 314.6, 287.4, 32627, 0, 1734544179492, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_Launch_v2_0', 3, 'obj2', 109.6, 293.4, 61068, 0, 1734544179492, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_Launch_v2_1', 1, 'obj2', 137.6, 238.4, 17496, 1, 1734544095466, 1),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_Launch_v2_2', 1, 'obj2', 525.6, 474.4, 16200, 1, 1734544300466, 1),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_Launch_v2_3', 1, 'obj3', 222.6, 290.4, 21710, 1, 1734543985773, 1),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_Launch_v2_4', 1, 'obj1', 470.6, 307.4, 11554, 0, 1734544267125, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_Launch_v2_4', 2, 'obj3', 315.6, 207.4, 36599, 0, 1734544267125, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_Launch_v2_4', 3, 'obj2', 382.6, 288.4, 76156, 0, 1734544267125, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_SoClose_0', 1, 'obj3', 57.6, 484.4, 6377, 1, 1734544032917, 1),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_SoClose_1', 1, 'obj2', 473.6, 359.4, 14608, 1, 1734544012172, 1),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_SoClose_2', 1, 'obj1', 169.6, 454.4, 11085, 1, 1734544053925, 1),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_SoClose_3', 1, 'obj1', 172.6, 335.4, 19325, 1, 1734544333293, 1),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_SoClose_4', 1, 'obj3', 467.6, 369.4, 10625, 1, 1734543945368, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_SoClose_4', 2, 'obj1', 230.6, 544.4, 21287, 1, 1734543945368, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Mech_SoClose_4', 3, 'obj1', 469.6, 377.4, 45024, 1, 1734543945368, 1),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Playground', 1, 'obj3', 524.6, 519.4, 7446, 0, 1734543720203, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Playground', 2, 'obj1', 48.6, 452.4, 51093, 0, 1734543720203, 0),
('5dfedfbb557425b599c8c523', '1761bed709be52efb17ea7e2f1a46af6', 'Playground', 3, 'obj1', 525.6, 545.4, 62804, 0, 1734543720203, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Basic_1', 1, 'obj1', 189, 342, 8603, 1, 1734542941082, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Basic_2', 1, 'obj3', 402, 233, 13153, 1, 1734542975195, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Comp_GapCatapult', 1, 'obj2', 359, 569, 33102, 1, 1734543462474, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Comp_GapCatapult', 2, 'obj1', 467, 578, 45503, 1, 1734543462474, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Comp_GapCatapult', 3, 'obj2', 141, 432, 53863, 1, 1734543462474, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Comp_SoCloseLaunch', 1, 'obj3', 165, 298, 8299, 0, 1734543536701, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Comp_SoCloseLaunch', 2, 'obj2', 292, 586, 39476, 0, 1734543536701, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Comp_SoCloseLaunch', 3, 'obj1', 232, 276, 62677, 0, 1734543536701, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Comp_UnboxSlope', 1, 'obj1', 262, 412, 8403, 1, 1734543395758, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Comp_UnboxSlope', 2, 'obj2', 313, 404, 31452, 1, 1734543395758, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_Launch_v2_0', 1, 'obj1', 78, 570, 6557, 1, 1734543115010, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_Launch_v2_1', 1, 'obj3', 131, 573, 12603, 1, 1734543237736, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_Launch_v2_2', 1, 'obj2', 535, 577, 4050, 1, 1734543344245, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_Launch_v2_2', 2, 'obj1', 545, 571, 8874, 1, 1734543344245, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_Launch_v2_3', 1, 'obj1', 132, 573, 6008, 1, 1734543187349, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_Launch_v2_4', 1, 'obj1', 483, 570, 5687, 1, 1734543214823, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_Launch_v2_4', 2, 'obj3', 490, 557, 19727, 1, 1734543214823, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_SoClose_0', 1, 'obj1', 47, 493, 7519, 1, 1734543300554, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_SoClose_0', 2, 'obj3', 45, 505, 12304, 1, 1734543300554, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_SoClose_1', 1, 'obj3', 476, 386, 16727, 1, 1734543147876, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_SoClose_2', 1, 'obj3', 136, 471, 13555, 1, 1734543172637, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_SoClose_3', 1, 'obj1', 167, 348, 3542, 1, 1734543319075, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Mech_SoClose_4', 1, 'obj2', 482, 382, 11014, 1, 1734543264233, 1),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Playground', 1, 'obj1', 425, 517, 4399, 0, 1734542913697, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Playground', 2, 'obj2', 572, 385, 19919, 0, 1734542913697, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Playground', 3, 'obj3', 562, 470, 30703, 0, 1734542913697, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Playground', 4, 'obj3', 520, 190, 35640, 0, 1734542913697, 0),
('5f47e34858dd331165bf9f00', '2441e9efe570b0c1e5125a549c272d2b', 'Playground', 5, 'obj1', 352, 553, 48648, 0, 1734542913697, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 1, 'obj2', 380, 187, 20651, 1, 1734543822023, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 2, 'obj3', 479, 91, 36191, 1, 1734543822023, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 3, 'obj2', 480, 103, 47135, 1, 1734543822023, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 4, 'obj1', 384, 188, 66348, 1, 1734543822023, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 5, 'obj3', 441, 175, 170252, 1, 1734543822023, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 6, 'obj2', 429, 180, 183524, 1, 1734543822023, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 7, 'obj1', 217, 212, 197587, 1, 1734543822023, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 8, 'obj1', 125, 187, 225331, 1, 1734543822023, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 9, 'obj3', 142, 182, 233883, 1, 1734543822023, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 10, 'obj2', 172, 192, 244410, 1, 1734543822023, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 11, 'obj3', 152, 188, 254002, 1, 1734543822023, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 12, 'obj1', 218, 218, 264126, 1, 1734543822023, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_1', 13, 'obj1', 178, 191, 292725, 1, 1734543822023, 1),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Basic_2', 1, 'obj3', 402, 236, 20513, 1, 1734543506504, 1),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Comp_GapCatapult', 1, 'obj1', 490, 526, 9782, 0, 1734544483575, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Comp_GapCatapult', 2, 'obj2', 481, 541, 19462, 0, 1734544483575, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Comp_GapCatapult', 3, 'obj3', 489, 545, 47033, 0, 1734544483575, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Comp_SoCloseLaunch', 1, 'obj2', 358, 434, 15706, 0, 1734544428323, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Comp_SoCloseLaunch', 2, 'obj3', 429, 432, 25246, 0, 1734544428323, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Comp_SoCloseLaunch', 3, 'obj3', 427, 424, 38240, 0, 1734544428323, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Comp_UnboxSlope', 1, 'obj2', 371, 418, 9748, 0, 1734544534352, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Comp_UnboxSlope', 2, 'obj3', 407, 396, 26261, 0, 1734544534352, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Comp_UnboxSlope', 3, 'obj1', 426, 449, 40866, 0, 1734544534352, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_Launch_v2_0', 1, 'obj2', 185, 234, 10587, 0, 1734544208338, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_Launch_v2_0', 2, 'obj3', 99, 246, 33680, 0, 1734544208338, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_Launch_v2_0', 3, 'obj1', 149, 232, 44901, 0, 1734544208338, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_Launch_v2_1', 1, 'obj3', 127, 187, 18451, 1, 1734544083101, 1),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_Launch_v2_2', 1, 'obj3', 419, 400, 12435, 0, 1734543968864, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_Launch_v2_2', 2, 'obj3', 416, 400, 38627, 0, 1734543968864, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_Launch_v2_2', 3, 'obj2', 540, 406, 53256, 0, 1734543968864, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_Launch_v2_3', 1, 'obj3', 226, 295, 10009, 1, 1734543859792, 1),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_Launch_v2_4', 1, 'obj2', 400, 265, 10201, 0, 1734544368669, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_Launch_v2_4', 2, 'obj2', 398, 273, 19361, 0, 1734544368669, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_Launch_v2_4', 3, 'obj2', 389, 278, 35120, 0, 1734544368669, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_SoClose_0', 1, 'obj1', 46, 495, 13505, 0, 1734544318378, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_SoClose_0', 2, 'obj3', 62, 503, 28289, 0, 1734544318378, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_SoClose_0', 3, 'obj2', 26, 506, 38838, 0, 1734544318378, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_SoClose_1', 1, 'obj3', 508, 422, 9836, 0, 1734544150029, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_SoClose_1', 2, 'obj1', 481, 392, 28029, 0, 1734544150029, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_SoClose_1', 3, 'obj1', 443, 394, 46864, 0, 1734544150029, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_SoClose_2', 1, 'obj2', 115, 473, 11525, 1, 1734544270952, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_SoClose_2', 2, 'obj3', 132, 480, 21095, 1, 1734544270952, 1),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_SoClose_3', 1, 'obj1', 136, 362, 12923, 1, 1734544236269, 1),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_SoClose_4', 1, 'obj2', 501, 434, 8494, 0, 1734544037227, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_SoClose_4', 2, 'obj3', 516, 429, 26963, 0, 1734544037227, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Mech_SoClose_4', 3, 'obj1', 516, 404, 53173, 0, 1734544037227, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Playground', 1, 'obj3', 247, 156, 12021, 0, 1734543443464, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Playground', 2, 'obj3', 456, 84, 29202, 0, 1734543443464, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Playground', 3, 'obj2', 450, 65, 52670, 0, 1734543443464, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Playground', 4, 'obj1', 68, 442, 85648, 0, 1734543443464, 0),
('6693df690cbd52bc4cbf870d', '8e40c9b8616eb0259bbeac287a7cb2e0', 'Playground', 5, 'obj3', 208, 156, 106952, 0, 1734543443464, 0),
('null', '29fa32b6300d83c1241ed11d41061813', 'Comp_GapCatapult', 1, 'obj2', 488, 577, 15266, 1, 1734543285298, 0),
('null', '29fa32b6300d83c1241ed11d41061813', 'Comp_GapCatapult', 2, 'obj2', 330, 574, 42065, 1, 1734543285298, 0),
('null', '29fa32b6300d83c1241ed11d41061813', 'Comp_GapCatapult', 3, 'obj2', 147, 412, 68017, 1, 1734543285298, 1),
('null', '29fa32b6300d83c1241ed11d41061813', 'Comp_SoCloseLaunch', 1, 'obj3', 165, 297, 23227, 1, 1734543206847, 0),
('null', '29fa32b6300d83c1241ed11d41061813', 'Comp_SoCloseLaunch', 2, 'obj2', 251, 265, 44724, 1, 1734543206847, 0),
('null', '29fa32b6300d83c1241ed11d41061813', 'Comp_SoCloseLaunch', 3, 'obj3', 260, 275, 56079, 1, 1734543206847, 1),
('null', '29fa32b6300d83c1241ed11d41061813', 'Comp_UnboxSlope', 1, 'obj1', 401, 328, 20290, 1, 1734543329356, 0),
('null', '29fa32b6300d83c1241ed11d41061813', 'Comp_UnboxSlope', 2, 'obj2', 299, 415, 31723, 1, 1734543329356, 1),
('null', '29fa32b6300d83c1241ed11d41061813', 'Mech_Launch_v2_0', 1, 'obj1', 78, 571, 4198, 1, 1734543086624, 1),
('null', '29fa32b6300d83c1241ed11d41061813', 'Mech_Launch_v2_1', 1, 'obj3', 134, 568, 3638, 1, 1734543114722, 1),
('null', '29fa32b6300d83c1241ed11d41061813', 'Mech_Launch_v2_3', 1, 'obj1', 134, 399, 4003, 1, 1734543056587, 1),
('null', '29fa32b6300d83c1241ed11d41061813', 'Mech_Launch_v2_4', 1, 'obj1', 479, 564, 4808, 1, 1734542979259, 1),
('null', '29fa32b6300d83c1241ed11d41061813', 'Mech_SoClose_0', 1, 'obj3', 51, 482, 9433, 1, 1734543015008, 1),
('null', '29fa32b6300d83c1241ed11d41061813', 'Mech_SoClose_1', 1, 'obj2', 472, 395, 7044, 1, 1734543074600, 1),
('null', '29fa32b6300d83c1241ed11d41061813', 'Mech_SoClose_2', 1, 'obj2', 149, 450, 9454, 1, 1734543044469, 0),
('null', '29fa32b6300d83c1241ed11d41061813', 'Mech_SoClose_2', 2, 'obj2', 158, 463, 15879, 1, 1734543044469, 0),
('null', '29fa32b6300d83c1241ed11d41061813', 'Mech_SoClose_2', 3, 'obj1', 148, 464, 20227, 1, 1734543044469, 1),
('null', '29fa32b6300d83c1241ed11d41061813', 'Mech_SoClose_3', 1, 'obj1', 168, 367, 4958, 1, 1734543131443, 1),
('null', '29fa32b6300d83c1241ed11d41061813', 'Playground', 1, 'obj1', 542, 216, 5565, 0, 1734542880219, 0),
('null', '29fa32b6300d83c1241ed11d41061813', 'Playground', 2, 'obj1', 545, 424, 23695, 0, 1734542880219, 0),
('null', '29fa32b6300d83c1241ed11d41061813', 'Playground', 3, 'obj1', 539, 123, 31831, 0, 1734542880219, 0),
('null', '29fa32b6300d83c1241ed11d41061813', 'Playground', 4, 'obj3', 147, 557, 40915, 0, 1734542880219, 0);

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
