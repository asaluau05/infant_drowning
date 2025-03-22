-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Jan 09, 2025 at 09:29 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `infant_drowning`
--

-- --------------------------------------------------------

--
-- Table structure for table `id_admin`
--

CREATE TABLE `id_admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `id_admin`
--

INSERT INTO `id_admin` (`username`, `password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `id_caretaker`
--

CREATE TABLE `id_caretaker` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `childname` varchar(20) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `status` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `id_caretaker`
--

INSERT INTO `id_caretaker` (`id`, `name`, `mobile`, `childname`, `uname`, `status`) VALUES
(1, 'Ram', 9894442716, 'Raj', 'girish', 1),
(2, 'Mani', 9677874082, 'Muthu', 'girish', 0);

-- --------------------------------------------------------

--
-- Table structure for table `id_register`
--

CREATE TABLE `id_register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL,
  `caretaker_mobile` bigint(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `id_register`
--

INSERT INTO `id_register` (`id`, `name`, `mobile`, `email`, `uname`, `pass`, `caretaker_mobile`) VALUES
(1, 'Girish', 9894442716, 'girish@gmail.com', 'girish', '123456', 0);
