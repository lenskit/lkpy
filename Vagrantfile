# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.provider "hyperv" do |hv|
    hv.maxmemory = 4096
    hv.cpus = 2
  end
  config.vm.define "centos" do |b|
    b.vm.box = "generic/centos8"
    b.vm.provision "shell", inline: <<-SHELL
      yum install -y python3-devel make
    SHELL
  end

  config.vm.define "alpine" do |b|
    b.vm.box = "generic/alpine312"
    b.vm.provision "shell", inline: <<-SHELL
      apk update
      apk add python3-dev make gcc g++ gfortran
    SHELL
  end

  config.vm.define "debian" do |b|
    b.vm.box = "generic/debian10"
    b.vm.provision "shell", inline: <<-SHELL
      apt update
      apt install -y python3-dev build-essential
    SHELL
  end

  config.vm.define "arch" do |b|
    b.vm.box = "generic/arch"
    b.vm.provision "shell", inline: <<-SHELL
      pacman -S --noconfirm python gcc-fortran
    SHELL
  end

  config.vm.define "freebsd" do |b|
    b.vm.box = "generic/freebsd12"
    b.vm.provision "shell", inline: <<-SHELL
      pkg update
      pkg install -y python37 py37-numpy py37-pandas py37-numba py37-scipy py37-scikit-learn
    SHELL
  end

  # Share an additional folder to the guest VM. The first argument is
  # the path on the host to the actual folder. The second argument is
  # the path on the guest to mount the folder. And the optional third
  # argument is a set of non-required options.
  # config.vm.synced_folder "../data", "/vagrant_data"

  # Provider-specific configuration so you can fine-tune various
  # backing providers for Vagrant. These expose provider-specific options.
  # Example for VirtualBox:
  #
  # config.vm.provider "virtualbox" do |vb|
  #   # Display the VirtualBox GUI when booting the machine
  #   vb.gui = true
  #
  #   # Customize the amount of memory on the VM:
  #   vb.memory = "1024"
  # end
  #
  # View the documentation for the provider you are using for more
  # information on available options.

  # Enable provisioning with a shell script. Additional provisioners such as
  # Ansible, Chef, Docker, Puppet and Salt are also available. Please see the
  # documentation for more information about their specific syntax and use.
  # config.vm.provision "shell", inline: <<-SHELL
  #   apt-get update
  #   apt-get install -y apache2
  # SHELL
end
