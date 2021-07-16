!Answer for Qn3 part 2
!Suppose the domain is split up into an nxn grid and the domain decomposition is
!m^2 rectangles. Each rectangle represents a process. Then there are (m^2-1)*n boundary
!points. Each of the boundary points has values of Clocal that need to be exchanged
!with the neighbouring process for it to be used during the iteration.
!Considering the second approach, the domain is split up into an nxn grid and
!the domain is decomposed into a square grid with m^2 squares each representing a
!process. Then there are 2*(m-1)*n boundary points across which information has
!to be exchanged. The smaller squares approach means less communication between
!the processes are needed, so communication is minimized. This makes the code
!quicker. However, decomposing the domain this way is harder to implement and
!requires tools such as MPI_cart_create and MPI_cart_shift to manage more complex
!topologies.

!In summary: Advantage: code run quicker, Disadvantage: harder to implement

!-------------------------------
!Fortran program for simulating contamination
!model using AOS method combined with a
!distributed memory approach (with MPI)
!The main program 1) Initializes MPI , 2) Reads in
!numerical and model parameters from data.in (which must be created),
!3) calls simulate_mpi (which must be completed) and 4) writes
! the final concentration field to a file that can be loaded witn numpy
!
! simulate_mpi: Sets up domain decomposition allocating different radial segments
! of the domain to different processes. Then each process uses AOS iteration
! to solve the model communicating only as necessary.
! After completing iterations, the routine gathers the radial portions of
! the final concentration fields from each process onto process 0 and this
! full field is returned to the main program

! To compile: mpif90 -o p3mpi.exe p3_dev.f90
! To run: mpiexec -n 2 p3mpi.exe

!----------------
program part3_mpi
    use mpi
    implicit none

    integer :: n! number of grid points (corresponds to (n+2 x n+2) grid)
    integer :: kmax !max number of iterations
    real(kind=8) :: tol, g !convergence tolerance, death rate
    real(kind=8), allocatable, dimension(:) :: deltac
    real(kind=8), allocatable, dimension(:,:) :: C !concentration matrix
    real(kind=8) :: S0,r0,t0 !source parameters
    real(kind=8) :: k_bc !r=1 boundary condition parameter
    integer :: i1,j1
    integer :: myid, numprocs, ierr

 ! Initialize MPI
    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

!gather input

    open(unit=10,file='data.in')
        read(10,*) n
        read(10,*) kmax
        read(10,*) tol
        read(10,*) g
        read(10,*) S0
        read(10,*) r0
        read(10,*) t0
        read(10,*) k_bc
   close(10)

    allocate(C(0:n+1,0:n+1),deltaC(kmax))

!compute solution
    call simulate_mpi(MPI_COMM_WORLD,numprocs,n,kmax,tol,g,S0,r0,t0,k_bc,C,deltaC)


!output solution (after completion of gather in euler_mpi)
     call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      if (myid==0) then
        open(unit=11,file='f.dat')
        do i1=0,n+1
            write(11,('(1000E28.16)')) C(i1,:)
        end do
        close(11)

        open(unit=12,file='deltac.dat')
        do i1=1,kmax
	       write(12,*) deltac(i1)
	      end do
        close(12)
      end if
    !can be loaded in python with: c=np.loadtxt('C.dat')

    call MPI_FINALIZE(ierr)
end program part3_mpi


!Simulate contamination model with AOS iteration
subroutine simulate_mpi(comm,numprocs,n,kmax,tol,g,S0,r0,t0,k_bc,C,deltaC)
    !input:
    !comm: MPI communicator
    !numprocs: total number of processes
    !n: total number of grid points in each direction (actually n+2 points in each direction)
    !tol: convergence criteria
    !kmax: total number of iterations
    !g: bacteria death rate
    !S0,r0,t0: source function parameters
    !k_bc: r=1 boundary condition parameter
    !output: C, final solution
    !deltaC: |max change in C| each iteration
    use mpi
    implicit none
    integer, intent (in) :: comm,numprocs,n,kmax
    real(kind=8), intent(in) ::tol,g,S0,r0,t0,k_bc
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: C
    real(kind=8), intent(out) :: deltac(kmax)
    integer :: i1,i2,j0,j1,k,istart,iend
    integer :: myid,ierr,nlocal,request
    real(kind=8) :: t1d(0:n+1),del,pi, dlocal
    real(kind=8), allocatable, dimension(:) :: r1d
    real(kind=8),allocatable, dimension(:,:) :: r,t,Clocal,rinv2, Sdel2,Cnew,fac,fac2,facp,facm
    integer, dimension(numprocs) :: disps,Nper_proc
    !real(kind=8), dimension(0:nlocal+1,0:n+1) ::rinv2, Sdel2,Cnew,fac,fac2,facp,facm
    integer, dimension(MPI_STATUS_SIZE) :: status


    call MPI_COMM_RANK(comm, myid, ierr)
    print *, 'start simulate_mpi, myid=',myid

    !Set up theta, goes around circle
    pi = acos(-1.d0)
    del = pi/dble(n+1)
    do i1 = 0,n+1
      t1d(i1) = i1*del
    end do

    !generate decomposition and allocate sub-domain variables
    !Note: this decomposition is for indices 1:n; it
    !ignores the boundaries at i=0 and i=n+1 which
    !should be assigned to the first and last processes
    call mpe_decomp1d(n,numprocs,myid,istart,iend)
    print *, 'istart,iend,threadID=',istart,iend,myid
    nlocal = iend-istart+1
    allocate(r1d(nlocal+2))

    !generate local grid and allocate Clocal----
    do i1=istart-1,iend+1
      r1d(i1-istart+2) = 1.d0 + i1*del
    end do

    allocate(r(0:nlocal+1,0:n+1),t(0:nlocal+1,0:n+1),Clocal(0:nlocal+1,0:n+1))
    do j1=0,n+1
      r(:,j1) = r1d
    end do
    do i1=0,nlocal+1
      t(i1,:) = t1d
    end do
  !  -----------------

    !Update equation parameters------
    !del2f = 0.25d0*(del**2)

    !Size of rinv2, Sdel2,Cnew,fac,fac2,facp,facm depends on nlocal which is calculated
    !in the code.
    !Allocated dimensions to allocate rinv2, Sdel2,Cnew,fac,fac2,facp,facm
    allocate(rinv2(0:nlocal+1,0:n+1),Sdel2(0:nlocal+1,0:n+1),Cnew(0:nlocal+1,0:n+1))
    allocate(fac(0:nlocal+1,0:n+1),fac2(0:nlocal+1,0:n+1),facp(0:nlocal+1,0:n+1))
    allocate(facm(0:nlocal+1,0:n+1))

    !Update equation parameters
    rinv2 = 1.d0/(r**2)
    fac = 1.d0/(2.d0+2.d0*rinv2+del*del*g) !1/(del^2*k^2 + 2 + 2/r^2)
    facp = (1.d0 + 0.5d0*del/r)*fac !(1 + del/(2r))*fac
    facm = (1.d0 - 0.5d0*del/r)*fac !(1 - del/(2r))*fac
    fac2 = fac*rinv2
    !-----------------

    !set initial condition/boundary conditions
    Clocal = (sin(k_bc*t)**2)*(pi+1.d0-r)/pi

    !set source function, Sdel2 = S*del^2*fac
    Sdel2 = s0*exp(-20.d0*((r-r0)**2+(t-t0)**2))*(del**2)*fac

    !Initiaize Cnew for it to be used in the iterations
    Cnew = Clocal

    !Intialize deltac for it to be used in the iterations
    deltac=0.d0

    !AOS iteration---
    do k = 1,kmax
      !mpi send proc myid to proc myid+1 Clocal[nlocal,:]
      !mpi recieve into Clocal(0,:)
      !mpi send myid to my id -1 Clocal(1,:)
      !mpi recieve into Clocal(nlocal+1,:)
      !send and recieve values up processor

      !Send boundary values of Clocal to neighbouring above processor for it to use in iteration
      if (myid/=numprocs-1) then
        call MPI_ISEND(Clocal(nlocal,:),n+2,MPI_DOUBLE_PRECISION,myid+1,0,MPI_COMM_WORLD,request,ierr)
      end if

      !Recieve boundary values of Clocal from neighbouring below processor to use in iteration
      if (myid/=0) then
        call MPI_RECV(Clocal(0,:),n+2,MPI_DOUBLE_PRECISION,myid-1,MPI_ANY_TAG,MPI_COMM_WORLD,status,ierr)
      end if

      !Send boundary values of Clocal to neighbouring below processor for it to use in iteration
      if (myid/=0) then
        call MPI_ISEND(Clocal(1,:),n+2, MPI_DOUBLE_PRECISION,myid-1,0,MPI_COMM_WORLD,request,ierr)
      end if

      !Recieve boundary values of Clocal from neighbouring above processor to use in iteration
      if (myid/=numprocs-1) then
        call MPI_RECV(Clocal(nlocal+1,:),n+2, MPI_DOUBLE_PRECISION,myid+1,MPI_ANY_TAG,MPI_COMM_WORLD,status,ierr)
      end if


      !Update all the whites
      do i1=1,nlocal
        !If i1 is odd then the block 1,3,5,7,... in that line in the chessboard need to be updated
        if (mod(istart+i1,2)==1) then
          do j1=1,n,2
            Cnew(i1,j1)=0.5d0*(-Clocal(i1,j1)+3.d0*(facm(i1,j1)*Clocal(i1-1,j1)+&
            fac2(i1,j1)*Clocal(i1,j1-1)+fac2(i1,j1)*Clocal(i1,j1+1)+&
            facp(i1,j1)*Clocal(i1+1,j1)+Sdel2(i1,j1)))
          end do
        !If i1 is even then the block 2,4,6,8,... in that line in the chessboard need to be updated
        else if (mod(istart+i1,2)==0) then
          do j1=2,n,2
            Cnew(i1,j1)=0.5d0*(-Clocal(i1,j1)+3.d0*(facm(i1,j1)*Clocal(i1-1,j1)+&
            fac2(i1,j1)*Clocal(i1,j1-1)+fac2(i1,j1)*Clocal(i1,j1+1)+&
            facp(i1,j1)*Clocal(i1+1,j1)+Sdel2(i1,j1)))
          end do
        end if
      end do

      !Send data across processors before updating blacks.
      !Update all blacks
      !Send boundary values of Clocal to neighbouring above processor for it to use in iteration
      if (myid/=numprocs-1) then
        call MPI_ISEND(Cnew(nlocal,:),n+2,MPI_DOUBLE_PRECISION,myid+1,0,MPI_COMM_WORLD,request,ierr)
      end if

      !Recieve boundary values of Clocal from neighbouring below processor to use in iteration
      if (myid/=0) then
        call MPI_RECV(Cnew(0,:),n+2,MPI_DOUBLE_PRECISION,myid-1,MPI_ANY_TAG,MPI_COMM_WORLD,status,ierr)
      end if

      !Send boundary values of Clocal to neighbouring below processor for it to use in iteration
      if (myid/=0) then
        call MPI_ISEND(Cnew(1,:),n+2, MPI_DOUBLE_PRECISION,myid-1,0,MPI_COMM_WORLD,request,ierr)
      end if

      !Recieve boundary values of Clocal from neighbouring above processor to use in iteration
      if (myid/=numprocs-1) then
        call MPI_RECV(Cnew(nlocal+1,:),n+2, MPI_DOUBLE_PRECISION,myid+1,MPI_ANY_TAG,MPI_COMM_WORLD,status,ierr)
      end if

      !Update blacks
      do i1=1,nlocal
        !If i1 is odd then the block 2,4,6,8,... in that line in the chessboard need to be updated
        if (mod(istart+i1,2)==1) then
          do j1=2,n,2
            Cnew(i1,j1)=0.5d0*(-Clocal(i1,j1)+3.d0*(facm(i1,j1)*Cnew(i1-1,j1)+&
            fac2(i1,j1)*Cnew(i1,j1-1)+fac2(i1,j1)*Cnew(i1,j1+1)+&
            facp(i1,j1)*Cnew(i1+1,j1)+Sdel2(i1,j1)))
          end do
        !If i1 is even then the block 1,3,5,7,... in that line in the chessboard need to be updated
        else if (mod(istart+i1,2)==0) then
          do j1=1,n,2
            Cnew(i1,j1)=0.5d0*(-Clocal(i1,j1)+3.d0*(facm(i1,j1)*Cnew(i1-1,j1)+&
            fac2(i1,j1)*Cnew(i1,j1-1)+fac2(i1,j1)*Cnew(i1,j1+1)+&
            facp(i1,j1)*Cnew(i1+1,j1)+Sdel2(i1,j1)))
          end do
        end if
      end do

      !Compute relative error
      dlocal = maxval(abs(Cnew(1:nlocal,1:n)-Clocal(1:nlocal,1:n)))

      !Choose maximum of all the dlocals in each process and reduce it down to deltac
      call MPI_ALLREDUCE(dlocal, deltac(k), 1, MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)

      !Set clocal to Cnew , update for it to be used in the next iteration
      Clocal(1:nlocal,1:n)=Cnew(1:nlocal,1:n)

      !If condition is met exit from iterations
      if (deltac(k)<tol) exit !check convergence criterion

      !Print statement every 1000 iterations
      if (myid==0) then
        if (mod(k,1000)==0) print *, k,deltac(k)
      end if

    end do
    !---------------



    !---------------------------------------------------------
    !Code below constructs C from the Clocal on each process
    print *, 'before collection',myid, maxval(abs(Clocal))

    i1=1
    i2 = nlocal

    if (myid==0) then
      i1=0
      nlocal = nlocal+1
    elseif (myid==numprocs-1) then
      i2 = nlocal+1
      nlocal = nlocal + 1
    end if

    call MPI_GATHER(nlocal,1,MPI_INT,NPer_proc,1,MPI_INT,0,comm,ierr)
    !collect Clocal from each processor onto myid=0

    if (myid==0) then
        disps(1)=0
        do j1=2,numprocs
          disps(j1) = disps(j1-1) + Nper_proc(j1-1)*(n+2)
        end do

        print *, 'nper_proc=',NPer_proc
        print *, 'disps=',disps
    end if

  !collect Clocal from each processor onto myid=0

     call MPI_GATHERV(transpose(Clocal(i1:i2,:)),nlocal*(n+2),MPI_DOUBLE_PRECISION,C,Nper_proc*(n+2), &
                 disps,MPI_DOUBLE_PRECISION,0,comm,ierr)

      C = transpose(C)
    if (myid==0) print *, 'finished',maxval(abs(C)),sum(C)


end subroutine simulate_mpi



!--------------------------------------------------------------------
!  (C) 2001 by Argonne National Laboratory.
!      See COPYRIGHT in online MPE documentation.
!  This file contains a routine for producing a decomposition of a 1-d array
!  when given a number of processors.  It may be used in "direct" product
!  decomposition.  The values returned assume a "global" domain in [1:n]
!
subroutine MPE_DECOMP1D( n, numprocs, myid, s, e )
    implicit none
    integer :: n, numprocs, myid, s, e
    integer :: nlocal
    integer :: deficit

    nlocal  = n / numprocs
    s       = myid * nlocal + 1
    deficit = mod(n,numprocs)
    s       = s + min(myid,deficit)
    if (myid .lt. deficit) then
        nlocal = nlocal + 1
    endif
    e = s + nlocal - 1
    if (e .gt. n .or. myid .eq. numprocs-1) e = n

end subroutine MPE_DECOMP1D
